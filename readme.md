# Less is more

This repository aims to provide the minimal functions for NMT training and be easy  and flexiable to adapt for research.

## Step 1. Train NMT Model 

```bash 

dump_path=checkpoint/ldc_nonorm_dp03_attndp01_5gpu/
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python mt_task.py train \
    --vocab_src '/data4/bjji/data/ldc/vocab.ch' \
    --vocab_tgt '/data4/bjji/data/ldc/vocab.en' \
    --train_src  '/data4/bjji/data/ldc/train.bpe.ch' \
    --train_tgt '/data4/bjji/data/ldc/train.bpe.en' \
    --valid_src '/data4/bjji/data/ldc/valid.bpe.ch' \
    --valid_tgt '/data4/bjji/data/ldc/valid.bpe.en' \
    --optim 'adam_inverse_sqrt,warmup_updates=4000,beta1=0.9,beta2=0.98,lr=0.0005' \
    --epoch_size 200000 \
    --batch_size 45 \
    --world_size 5 \
    --max_epoch 50 --seed 20 \
    --dump_path  $dump_path \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --dropout 0.3 \
    --attn_dropout 0.1

```

After training by several epochs(near 25 epochs), the model will converge and you can test these models by such command lines

## Step 2. Test NMT Model
```bash
tst_sets="nist02 nist03 nist04 nist05 nist08"

output=$dump_path
for tst in $tst_sets;do
CUDA_VISIBLE_DEVICES=2 python mt_task.py infer \
    --batch_size 30 \
    --beam 5 \
    --vocab_src '/data4/bjji/data/ldc/vocab.ch' \
    --vocab_tgt '/data4/bjji/data/ldc/vocab.en' \
    --reload_path $dump_path/checkpoint_best.pth \
    --src /data4/bjji/data/ldc/$tst.bpe.in > $output/$tst.out

 cat $output/$tst.out |grep 'Target_' | cut -f2- -d " " > $output/$tst.decoded
 sed -r -i 's/(@@ )|(@@ ?$)//g'  $output/$tst.decoded
 perl multi-bleu.perl   -lc /data4/bjji/data/ldc/$tst.ref.* < $output/$tst.decoded

done
```

## Results
***

|nist02 | nist03 | nist04 |nist05 |nist08 | avg|
|---|---- | --- | ---| ---    | ---   |
|fariseq| 47.16| 46.16 | 47.09|46.33    | 38.11   | 44.93 |
|simpleNMT| 47.80| 46.08 | 47.57|46.97    | 36.55   | 44.99 |

