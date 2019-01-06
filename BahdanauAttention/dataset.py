import torch
import numpy as np
from torch.utils.data import Dataset

## ============================== packing and padding ==============================
## ============================== length is needed ==============================
def collate_fn(insts):
    # if seq_pad in class then all seqs with same length
    maxlen = max([len(x) for x in insts])
    #maxlen = 24
    seq = np.array([x + [0] * (maxlen - len(x)) for x in insts])
    seq_lens = np.array([len(x) for x in insts])
    return torch.LongTensor(seq), torch.LongTensor(seq_lens)

def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = zip(*seq_pairs)
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

class Fra2EngDatasets(Dataset):
    def __init__(self, src, tgt):
        # self.device = device
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
