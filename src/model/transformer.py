
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append("..")
from ..utils.labelsmoothing import LabelSmoothingLoss

MAX_POSITIONS = 512

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.xavier_uniform_(m.weight)
    # if bias:
    #     nn.init.constant_(m.bias, 0.0)
    return m

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, \
                dropout=0.3, attention_dropout=0.1, activation='relu', src_dictionary=None, tgt_dictionary=None):
        
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dropout, attention_dropout, activation, src_dictionary=src_dictionary)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dropout, attention_dropout, activation, tgt_dictionary=tgt_dictionary)
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary

    def forward(self, mode, **kwargs):
        if mode == 'fwd':
            return self.fwd(**kwargs)
        else:
            return self.predict(**kwargs)

    def fwd(self, src, tgt):
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out)
        return decoder_out
    
    def predict(self, tensor, y):
        return self.decoder.predict(tensor, y)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                 dropout=0.3, attention_dropout=0.1, activation='relu', src_dictionary=None):

        super(TransformerEncoder, self).__init__()
        self.dictionary = src_dictionary
        self.embedding = Embedding(len(self.dictionary), d_model, padding_idx = self.dictionary.pad_index)
        self.num_layers = num_encoder_layers
        self.layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=attention_dropout, activation=activation),
            num_layers = self.num_layers,
            )
        self.position_embedding = Embedding(MAX_POSITIONS, d_model)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-12)
        self.dropout = dropout

    def forward(self, src):
        """
        src: [s, n]
        mask: [s, s]
        src_key_padding_mask:[n, s]
        output: [s, n, dim]
        """
        embed = self.embedding(src) 
        position = torch.arange(src.size(0)).unsqueeze(0).to(src.device).t()
        pos_embed = self.position_embedding(position)
        embed = embed + pos_embed.expand_as(embed) 
        embed = self.layer_norm(embed)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        src_key_padding_mask = (src == self.dictionary.pad_index)
        output = self.layer(embed, src_key_padding_mask=src_key_padding_mask.t())
        return {
                'encoder_out':output,
                'src_key_padding_mask':src_key_padding_mask.t()
        }


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 dropout=0.3, attention_dropout=0.1, activation='relu', tgt_dictionary=None):
        
        super(TransformerDecoder, self).__init__()
        self.dictionary = tgt_dictionary
        self.embedding = Embedding(len(self.dictionary), d_model, padding_idx=self.dictionary.pad_index)
        self.num_layers = num_decoder_layers
        self.layer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=attention_dropout, activation=activation),
            num_layers = self.num_layers,
        )
        self.output_layer = Linear(d_model, len(self.dictionary), bias=True)
        self.d_model = d_model
        self.position_embedding = Embedding(MAX_POSITIONS, d_model)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-12)
        self.dropout = dropout
        self.loss_fn = LabelSmoothingLoss(len(tgt_dictionary), 0.1)

    def forward(self, tgt, encoder_out):
        """
        tgt: [t, n]
        """
        embed = self.embedding(tgt) 
        position = torch.arange(tgt.size(0)).unsqueeze(0).to(tgt.device).t()
        pos_embed = self.position_embedding(position)
        embed = embed + pos_embed.expand_as(embed) 
        embed = self.layer_norm(embed)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        tgt_len = (tgt!=self.dictionary.pad_index).sum(dim=0)
        tgt_mask = (torch.triu(torch.ones(tgt_len.max(), tgt_len.max())) != 1).transpose(0, 1).to(embed.device)
        tgt_key_padding_mask = (tgt == self.dictionary.pad_index)
        output = self.layer(embed, memory=encoder_out['encoder_out'], tgt_mask=tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=tgt_key_padding_mask.t(), \
                            memory_key_padding_mask=encoder_out['src_key_padding_mask'])
        return output


    def predict(self, tensor, y):
        no_pad_mask = y != self.dictionary.pad_index
        y = y[no_pad_mask]
        no_masked_tensor = tensor[no_pad_mask.unsqueeze(-1).expand_as(tensor)].view(-1,self.d_model)
        scores = self.output_layer(no_masked_tensor).view(-1, len(self.dictionary))
        loss = self.loss_fn(scores, y)
        return loss, scores   

    def generate_greedy(self, encoder_out, tgt_langid=None, max_len=200):
        
        bsz = encoder_out['encoder_out'].size(1)
        generated = torch.LongTensor(max_len,bsz).fill_(self.dictionary.pad_index).to(encoder_out['src_key_padding_mask'].device)
        cur_len = 1
        unfinished = generated.new(bsz).long().fill_(1)
        generated[0] = self.dictionary.bos_index if not tgt_langid else tgt_langid
        while cur_len < max_len:
            tensor = self.forward(generated[:cur_len], encoder_out) # [s, n , dim]
            tensor = tensor[-1,:,:]                                                     # [1, n, dim]
            scores = self.output_layer(tensor).squeeze(0) # [n, nwords]
            next_words = scores.topk(1, -1)[1].squeeze(1)
            generated[cur_len] = next_words * unfinished + self.dictionary.pad_index *(1-unfinished)
            unfinished.mul_((next_words.ne(self.dictionary.eos_index)).long()) 
            cur_len += 1
            if unfinished.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished.byte(), self.dictionary.eos_index) 
        return generated.t()

    def generate_beam(self, encoder_out, tgt_langid=None, max_len=200, beam_size=1):
        
        bsz = encoder_out['encoder_out'].size(1)
        device = encoder_out['src_key_padding_mask'].device
        _expand_encoder_out(encoder_out, beam_size)
        
        generated = torch.LongTensor(max_len, bsz*beam_size).fill_(self.dictionary.pad_index).to(device)
        generated[0] = self.dictionary.bos_index if not tgt_langid else tgt_langid

        beam_scores = torch.FloatTensor(bsz, beam_size).fill_(0).to(device)
        beam_scores[:,1:] = -1e9  # assume first word bos always from first beam
        final_scores = beam_scores.clone()
        
        cur_len = 1
        done = [0 for _ in range(beam_size*bsz)]
        while cur_len < max_len-1:
            tensor = self.forward(generated[:cur_len], encoder_out) # [s, nxbeam, dim]
            tensor = tensor[-1,:,:]  #[1,nxbeam, dim]
            scores = torch.log_softmax(self.output_layer(tensor).squeeze(0), dim=-1) # [n x beam, nwords]
            scores = scores + beam_scores.view(-1,1)
            scores = scores.view(bsz, -1) # [n, beam x nwords]

            next_scores, next_words = torch.topk(scores, beam_size, dim=-1) # [n, beam]  
            n_words = len(self.dictionary)

            for sent_id in range(bsz):
                new_beam = []
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    beam_idx = idx // n_words
                    word_idx = idx % n_words
                    sent = generated[:cur_len, sent_id*beam_size+beam_idx].tolist() + [word_idx]
                    sent = torch.LongTensor(sent).to(device)
                    new_beam.append((sent, value))
                
                # update beam set
                step = 0
                for j in range(beam_size):
                    if done[sent_id*beam_size+j]: continue
                    generated[:cur_len+1, sent_id*beam_size+j].copy_(new_beam[step][0])
                    beam_scores[sent_id][j] = new_beam[step][1]
                    if new_beam[step][0][-1] == self.dictionary.eos_index:
                        done[sent_id*beam_size+j] = 1
                        beam_scores[sent_id, j] = -1e9
                        final_scores[sent_id, j] = value
                    step += 1
            cur_len += 1
            if all(done):
                break

        # add eos
        if not all(done):
            for i in range(len(done)):
                if not done[i]:
                    generated[-1,i] = self.dictionary.eos_index
                    final_scores[i%bsz, i%beam_size] = beam_scores[i%bsz, i%beam_size] # bsz, beam
        
        assert (generated==self.dictionary.eos_index).sum() == bsz * beam_size
        # final_scores = final_scores/generated.ne(self.dictionary.pad_index).sum(0).float().view(bsz, beam_size)**0.5
        indices = final_scores.topk(1,dim=-1)[1].view(-1) + torch.arange(bsz).to(device) * beam_size
        res = generated[:,indices].clone()
        return res.t()

def _expand_encoder_out(encoder_out, beam_size):
    seqlen = encoder_out['encoder_out'].size(0)
    dim = encoder_out['encoder_out'].size(2)
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1,1,beam_size,1).view(seqlen,-1,dim)
    encoder_out['src_key_padding_mask'] = encoder_out['src_key_padding_mask'].unsqueeze(1).\
                                          repeat(1,beam_size,1).view(-1,encoder_out['src_key_padding_mask'].size(-1))
