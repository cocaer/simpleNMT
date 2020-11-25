  
import json
import torch
import torch.utils.data as data
import bisect


class ParallelDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, trg_path, src_word2id, trg_word2id, special_bos=None):
        """Reads source and target sequences from txt files."""
        self.src_seqs = open(src_path).read().splitlines()
        self.trg_seqs = open(trg_path).read().splitlines()
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.pad_index = src_word2id.pad_index
        self.src_bos_index = src_word2id.bos_index
        self.tgt_bos_index = src_word2id.bos_index if special_bos else special_bos
        self.eos_index = src_word2id.eos_index
        assert len(src_seqs) == len(trg_seqs)
        assert src_word2id.pad_index == trg_word2id.pad_index and  src_word2id.bos_index == trg_word2id.bos_index \
               and src_word2id.eos_index == trg_word2id.eos_index
               
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        tokens = sequence.split()
        sequence = []
        bos_index = self.src_bos_index if not trg else self.tgt_bos_index
        sequence.append(bos_index)
        sequence.extend([word2id.index(token) for token in tokens])
        sequence.append(self.eos_index)
        sequence = torch.Tensor(sequence)
        return sequence

    def parallel_collate_fn(self, data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long().fill_(self.pad_index)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x[0]), reverse=True)

        src_seqs, trg_seqs = zip(*data)

        src_seqs, src_lengths = merge(src_seqs)
        trg_seqs, trg_lengths = merge(trg_seqs)

        return src_seqs, src_lengths, trg_seqs, trg_lengths
    
def get_train_parallel_loader(src_path, trg_path, src_word2id, trg_word2id, batch_size):
    dataset = ParallelDataset(src_path, trg_path, src_word2id, trg_word2id)
    sampler =  torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            collate_fn=dataset.parallel_collate_fn,
                                            sampler = sampler,
                                            )
    return data_loader

def get_valid_parallel_loader(src_path, trg_path, src_word2id, trg_word2id, batch_size):
    dataset = ParallelDataset(src_path, trg_path, src_word2id, trg_word2id)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=dataset.parallel_collate_fn)
    return data_loader
                      




class MultiParallelDataset(data.ConcatDataset):
    def __init__(self, langs, path_prefix, src_word2id, trg_word2id):
        super(MultiParallelDataset, self).__init__()
        self.datasets = _load(langs, path_prefix)
        

    @staticmethod
    def _load(langs, path_prefix, src_word2id, trg_word2id):
        datasets = []
        for step, prefix in  zip(langs, path_prefix):
            src, trg = step.split('-')
            src_path = f'{prefix}.{src}'
            trg_path = f'{prefix}.{trg}'
            assert f'<2{trg}>' in trg_word2id.word2id
            special_bos = trg_word2id.word2id['<2{trg}>']
            datasets.append(ParallelDataset(src_path, trg_path, src_word2id, trg_word2id, special_bos))
        return datasets












  

        