import os
import torch
import argparse
from src.data.vocabulary import  Dictionary
from src.data  import dataloader
from src.model.transformer import TransformerModel
from src.optim.optim import get_optimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils.logger import create_logger


def setup(rank, world_size, master_addr='localhost', master_port='12345'):
    os.environ['MASTER_ADDR'] = 'localhost'#master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def run_train(fn, args):
    mp.spawn(fn,
            args=(args, ),
            nprocs=args.world_size,
            join=True)

class Trainer(object):

    def __init__(self, model, data, args):

        super(Trainer, self).__init__()
        self.model = model
        self.data = data['dataloader']
        self.optimizer  = get_optimizer(self.model.parameters(), args.optim)
        self._best_valid_loss = float('inf')
        self.dump_path = args.dump_path
        self.world_size = args.world_size
        self.args = args
        self.logger = create_logger(os.path.join(self.dump_path,'train.log'), self.args.local_rank)
        if args.reload_path != "":
            self.load_checkpoint(args.reload_path)

    def mt_step(self, epoch):
        self.model.train()
        n_sentences = 0

        self.data['train'].sampler.set_epoch(epoch)
        while n_sentences < self.args.epoch_size:
            for i, batch in enumerate(iter(self.data['train'])):
                src, _, tgt, _ = batch
                y = tgt[:,:-1]    
                y_label = tgt[:,1:]
                src, y, y_label = src.cuda(), y.cuda(), y_label.cuda()
                decoder_out = self.model('fwd', src=src.t(), tgt=y.t())
                loss, _ = self.model('predict', tensor=decoder_out.transpose(0,1), y=y_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()   
                n_sentences += y_label.size(0)
             
                if n_sentences >= self.args.epoch_size:
                    break
             
                if i % 20 == 0:
                    self.logger.info(f"loss: {loss.item():.4f}")

    def load_checkpoint(self, path):
        
        if not os.path.isfile(path):
            return

        checkpoint_path = path
        data = torch.load(checkpoint_path, map_location='cpu')
        self._best_valid_loss = data['best_valid_loss']
        
        try:
            self.model.load_state_dict({k[len('module.'):] : data['module'][k]  for k in data['module']})
        except RuntimeError:
            self.model.load_state_dict({'module.'+ k : data['module'][k]  for k in data['module']})

        for group_id, param_group in enumerate(self.optimizer.param_groups):
            if 'num_updates' not in param_group:
                continue
            param_group['num_updates'] = data['optimizer']['param_groups'][group_id]['num_updates']
            param_group['lr'] = self.optimizer.get_lr_for_step(param_group['num_updates'])

        print(f"Load model from {path}")

    def save_checkpoint(self, name="", epoch=0):

        if self.args.local_rank != 0:
            return 

        data = {}
        checkpoint_path = os.path.join(self.dump_path, f"checkpoint_{name}.pth")
        data['optimizer'] = self.optimizer.state_dict()
        data['module'] = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        data['best_valid_loss'] = self._best_valid_loss
        data['epoch'] = epoch
        torch.save(data, checkpoint_path)

    def save_best_checkpoint(self, epoch, valid_loss):
        if valid_loss < self._best_valid_loss:
            self.save_checkpoint(f'best', epoch)
            self._best_valid_loss = valid_loss

    def evaluate(self, epoch):
        valid_loss = 0
        ntokens = 0.1
        self.model.eval()
       
        for i, batch in enumerate(iter(self.data['valid'])):
            src, _, tgt, _ = batch
            y = tgt[:,:-1]    
            y_label = tgt[:,1:]
            src, y, y_label = src.cuda(), y.cuda(), y_label.cuda()
            decoder_out = self.model('fwd', src=src.t(), tgt=y.t())
            loss, _ = self.model('predict', tensor=decoder_out.transpose(0,1), y=y_label)
            ntokens += y_label.size(1)
            valid_loss += loss.item() * y_label.size(1) 
       
        valid_loss /= ntokens
        self.logger.info(f"=============== Evaluation ==================")
        self.logger.info(f"loss on valid set: {valid_loss}")
        self.save_best_checkpoint(epoch, valid_loss)

def train(rank,  args):
    print(f"Running basic DDP example on rank {rank} {args.master_port}.")
    setup(rank, args.world_size,  args.master_port)
    args.local_rank = rank
    torch.manual_seed(args.seed)
    torch.cuda.set_device(rank)
    src_vocab = Dictionary.read_vocab(args.vocab_src)
    tgt_vocab = Dictionary.read_vocab(args.vocab_tgt)
    batch_size = args.batch_size
    
    # model init 
    model = TransformerModel(d_model=args.d_model, 
                            nhead=args.nhead, 
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers,
                            dropout=args.dropout,
                            attention_dropout=args.attn_dropout,
                            src_dictionary=src_vocab, 
                            tgt_dictionary=tgt_vocab)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        print(model)
    
    # data load
    train_loader = dataloader.get_train_parallel_loader(args.train_src, args.train_tgt, src_vocab, tgt_vocab,  batch_size=batch_size)
    valid_loader = dataloader.get_valid_parallel_loader(args.valid_src, args.valid_tgt, src_vocab, tgt_vocab,  batch_size=batch_size)

    data = {
        'dataloader': {'train': train_loader, 'valid':valid_loader}
    }


    trainer = Trainer(model, data,  args)
    for epoch in range(1,args.epoch_size):
        trainer.mt_step(epoch)
        trainer.evaluate(epoch)
        trainer.save_checkpoint(epoch)

def translate(args):
    batch_size = args.batch_size

    src_vocab = Dictionary.read_vocab(args.vocab_src)
    tgt_vocab = Dictionary.read_vocab(args.vocab_tgt)
    data = torch.load(args.reload_path, map_location='cpu')
    model = TransformerModel(src_dictionary=src_vocab, tgt_dictionary=tgt_vocab)
    model.load_state_dict({k : data['module'][k]  for k in data['module']})
    model.cuda()
    model.eval()
   
    if 'epoch' in data:
        print(f"Loading model from epoch_{data['epoch']}....")

    src_sent =  open(args.src, "r").readlines()
    for i in range(0, len(src_sent), batch_size):
        word_ids = [torch.LongTensor([src_vocab.index(w) for w in s.strip().split()]) 
                                                for s in src_sent[i:i+batch_size]]
        lengths = torch.LongTensor([len(s)+2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(src_vocab.pad_index)
        batch[0] = src_vocab.bos_index
       
        for j,s in enumerate(word_ids):
            if lengths[j] > 2:
                batch[1:lengths[j]-1,j].copy_(s)
            batch[lengths[j]-1,j] = src_vocab.eos_index
       
        batch = batch.cuda() 
        encoder_out = model.encoder(batch)
       
        with torch.no_grad():
            if args.beam == 1:
                generated = model.decoder.generate_greedy(encoder_out)
            else:
                generated = model.decoder.generate_beam(encoder_out, beam_size=5)
        
        for j, s in enumerate(src_sent[i:i+batch_size]):
            print(f"Source_{i+j}: {s.strip()}")
            hypo = []
            for w in generated[j][1:]:
                if tgt_vocab[w.item()] == '</s>':
                   break
                hypo.append(tgt_vocab[w.item()])
            hypo = " ".join(hypo)
            print(f"Target_{i+j}: {hypo}\n")


def init_exp(args):
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="train", choices=['train','infer'])
    

    # for train
    parser.add_argument("--vocab_src", type=str, default="")
    parser.add_argument("--vocab_tgt", type=str, default="")
    
    parser.add_argument("--train_src", type=str, default="")
    parser.add_argument("--train_tgt", type=str, default="")
    
    parser.add_argument("--valid_src", type=str, default="")
    parser.add_argument("--valid_tgt", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epoch_size", type=int, default=200000)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master_port", type=int, default=23451)
    parser.add_argument("--world_size", type=int, default=4)

    parser.add_argument("--dump_path", type=str, default="checkpoint/ldc")
    parser.add_argument("--reload_path", type=str, default="")
    parser.add_argument("--optim", type=str, default="adam_inverse_sqrt,warmup_updates=4000,beta1=0.9,beta2=0.98,lr=0.0005")

    # for model architechture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attn_dropout", type=float, default=0.1)


    # for inference
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--beam", type=int, default=5)

    args = parser.parse_args()
    print(args)
    init_exp(args)
    
    if args.mode == 'train': 
        run_train(train, args)
    else:
        translate(args)

