import torch
import vocabulary
import dataloader
import sys
sys.path.append("..")
from model.transformer import TransformerModel


vocab_src = '/data4/bjji/data/ldc/vocab.ch'
vocab_tgt = '/data4/bjji/data/ldc/vocab.en'

src_vocab = vocabulary.Dictionary.read_vocab(vocab_src)
tgt_vocab = vocabulary.Dictionary.read_vocab(vocab_tgt)

src_path = '/data4/bjji/data/ldc/test.bpe.ch'
tgt_path = '/data4/bjji/data/ldc/test.bpe.en'

data_loader = dataloader.get_loader(src_path, tgt_path, src_vocab, tgt_vocab, batch_size=3)

data_iter = iter(data_loader)

transformer = TransformerModel(src_dictionary=src_vocab, tgt_dictionary=tgt_vocab)
# import pdb; pdb.set_trace()
for i, batch in enumerate(data_iter):
    if i==4:
        break
    src, _, tgt, _ = batch
    encoder_out = transformer.encoder(src.t())
    tgt = tgt[:,:-1]    
    decoder_out = transformer.decoder(tgt.t(), encoder_out=encoder_out)
    import pdb; pdb.set_trace()
