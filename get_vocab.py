# cat /data4/bjji/data/ldc/train.bpe.ch  | python get_vocab.py  > vocab.ch

import sys
from collections import Counter

cnt = Counter()
for line in sys.stdin.readlines():
    words = line.rstrip().split()
    for w in words:
        cnt[w] += 1

res = cnt.most_common(len(cnt))

for k in res:
    print(f"{k[0]} {k[1]}")