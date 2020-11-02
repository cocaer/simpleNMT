from src.data.vocabulary import  Dictionary
from src.data  import dataloader
from src.model.transformer import TransformerModel
from src.optim.optim import get_optimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch

vocab_src = '/data4/bjji/data/ldc/vocab.ch'
vocab_tgt = '/data4/bjji/data/ldc/vocab.en'

src_vocab = Dictionary.read_vocab(vocab_src)
tgt_vocab = Dictionary.read_vocab(vocab_tgt)

src_path = '/data4/bjji/data/ldc/test.bpe.ch'
tgt_path = '/data4/bjji/data/ldc/test.bpe.en'

optim_args = "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005"
epoch_size = 50
batch_size = 20
seed = 100
world_size = 4
dump_path = "checkpoint/test"

def setup(gpu_id, ngpu, master_addr='localhost', master_port='12345'):
    os.environ['MASTER_ADDR'] = 'localhost'#master_addr
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=gpu_id, world_size=ngpu)


def run_train(fn, world_size):
    mp.spawn(fn,
            args=(world_size,'localhost','12345'),
            nprocs=world_size,
            join=True)



class Trainer(object):

    def __init__(self, model, data, args):
        super(Trainer, self).__init__()

        self.model = model
        self.data = data['dataloader']
        self.optimizer  = get_optimizer(self.model.parameters(), optim_args)
        self._best_valid_loss = float('inf')
        self.dump_path = args.dump_path
        self.world_size = args.world_size

        if args.reload_path != "":
            self.load_checkpoint(args.reload_path)

    def init_logger(self):
        pass 
    
    def mt_step(self):
        self.model.train()
        for i, batch in enumerate(iter(self.data['train'])):
            src, _, tgt, _ = batch
            y = tgt[:,:-1]    
            y_label = tgt[:,1:]
            src, y, y_label = src.cuda(), y.cuda(), y_label.cuda()
            decoder_out = self.model('fwd', src=src.t(), tgt=y.t())
            loss, _ = self.model('predict', tensor=decoder_out.transpose(0,1), y=y_label)
            print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

    def load_checkpoint(self, path):
        
        if not os.path.isfile(path):
            return

        checkpoint_path = path
        data = torch.load(checkpoint_path, map_location='cpu')

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

    def save_checkpoint(self, name):
        data = {}
        checkpoint_path = os.path.join(self.dump_path, f"checkpoint_{name}.pth")

        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

        data['optimizer'] = self.optimizer.state_dict()
        data['module'] = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        torch.save(data, checkpoint_path)


    def save_best_checkpoint(self, epoch, valid_loss):
        if valid_loss < self._best_valid_loss:
            self.save_checkpoint('best')
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
        
        print(f"ntokens:{ntokens}")
        valid_loss /= ntokens
        print(f"loss on valid set is {valid_loss}")
        self.save_best_checkpoint(epoch, valid_loss)




class meta:
    pass

def train(gpu_id, ngpu, master_addr, master_port):
    print(f"Running basic DDP example on rank {gpu_id} {master_port}.")
    setup(gpu_id, ngpu, master_addr, master_port)

    torch.manual_seed(seed)
    torch.cuda.set_device(gpu_id)
    

    # model init 
    model = TransformerModel(src_dictionary=src_vocab, tgt_dictionary=tgt_vocab)
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])


    # data load
    train_loader = dataloader.get_train_loader(src_path, tgt_path, src_vocab, tgt_vocab,  batch_size=batch_size, ngpu=ngpu, gpu_id=gpu_id)
    valid_loader = dataloader.get_valid_loader(src_path, tgt_path, src_vocab, tgt_vocab,  batch_size=batch_size)

    data = {
        'dataloader': {'train': train_loader, 'valid':valid_loader}
    }


    args = meta()
    args.epoch_size = epoch_size
    args.world_size = world_size
    args.dump_path = dump_path
    args.reload_path = ""

    trainer = Trainer(model, data,  args)

    for epoch in range(args.epoch_size):
        trainer.mt_step()
        trainer.evaluate(epoch)
        # trainer.save_checkpoint(epoch)

def translate():
    args = meta()
    args.reload_path = ""
    args.src = ""
    

if __name__ == "__main__":
    run_train(train, world_size)
