
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=6, dim_feedforward=2048, \
                dropout=0.1, activation='relu', src_dictionary=None, tgt_dictionary=None):
        
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, src_dictionary)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, src_dictionary)
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary

    def forward(self, src, tgt):
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out)
        return decoder_out
        

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', src_dictionary=None):

        super(TransformerEncoder, self).__init__()
        self.dictionary = src_dictionary
        self.embedding = nn.Embedding(len(self.dictionary), d_model, padding_idx = self.dictionary.pad_index)
        self.num_layers = num_encoder_layers
        self.layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation),
            num_layers = self.num_layers
        )

    def forward(self, src):
        """
        src: [s, n]
        mask: [s, s]
        src_key_padding_mask:[n, s]
        output: [s, n, dim]
        """
        src_key_padding_mask = (src == self.dictionary.pad_index).t()
        embed = self.embedding(src)
        output = self.layer(embed, src_key_padding_mask=src_key_padding_mask)
        return {
                'encoder_out':output,
                'src_key_padding_mask':src_key_padding_mask
        }


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', tgt_dictionary=None):
        
        super(TransformerDecoder, self).__init__()
        self.dictionary = tgt_dictionary
        self.embedding = nn.Embedding(len(self.dictionary), d_model, padding_idx=self.dictionary.pad_index)
        self.num_layers = num_decoder_layers
        self.layer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation),
            num_layers = self.num_layers
        )
        self.output_layer = nn.Linear(d_model, len(self.dictionary), bias=True)
        self.d_model = d_model

    def forward(self, tgt, encoder_out):
        """
        tgt: [t, n]
        """
        embed = self.embedding(tgt)
        tgt_len = (tgt!=self.dictionary.pad_index).sum(dim=0)
        tgt_mask = (torch.triu(torch.ones(tgt_len.max(), tgt_len.max())) != 1).transpose(0, 1).to(embed.device)
        tgt_key_padding_mask = (tgt == self.dictionary.pad_index).t()
        output = self.layer(embed, memory=encoder_out['encoder_out'], tgt_mask=tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=tgt_key_padding_mask, \
                            memory_key_padding_mask=encoder_out['src_key_padding_mask'])
        return output

    def compute_mt_loss(self, tensor, y):
        no_pad_mask = y != self.dictionary.pad_index
        y = y[no_pad_mask]
        masked_tensor = tensor[no_pad_mask.unsqueeze(-1).expand_as(tensor)].view(-1,self.d_model)
        scores = self.output_layer(masked_tensor).view(-1, len(self.dictionary))
        loss = F.cross_entropy(scores, y, reduction='mean')
        return loss, scores