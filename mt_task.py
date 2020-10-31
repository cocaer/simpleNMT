from src.data.vocabulary import  Dictionary
from src.data  import dataloader
from src.model.transformer import TransformerModel
from src.optim.optim import get_optimizer



vocab_src = '/data4/bjji/data/ldc/vocab.ch'
vocab_tgt = '/data4/bjji/data/ldc/vocab.en'

src_vocab = Dictionary.read_vocab(vocab_src)
tgt_vocab = Dictionary.read_vocab(vocab_tgt)

src_path = '/data4/bjji/data/ldc/test.bpe.ch'
tgt_path = '/data4/bjji/data/ldc/test.bpe.en'

optim_args = "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005"
epoch_size = 30
batch_size = 30

data_loader = dataloader.get_loader(src_path, tgt_path, src_vocab, tgt_vocab, batch_size=batch_size)
data_iter = iter(data_loader)
transformer = TransformerModel(src_dictionary=src_vocab, tgt_dictionary=tgt_vocab)
transformer.cuda()
optimizer = get_optimizer(transformer.parameters(), optim_args)




for i in range(epoch_size):

    for i, batch in enumerate(data_iter):
        src, _, tgt, _ = batch
        y = tgt[:,:-1]    
        y_label = tgt[:,1:]
        src, y, y_label = src.cuda(), y.cuda(), y_label.cuda()

        encoder_out = transformer.encoder(src.t())
        decoder_out = transformer.decoder(y.t(), encoder_out=encoder_out)
        loss, _ = transformer.decoder.compute_mt_loss(decoder_out.transpose(0,1), y_label)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()