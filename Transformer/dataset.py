import Constants
import numpy as np
import torch
from torch.utils.data import Dataset

def collate_fn(insts):
    # if seq_pad in class then all seqs with same length
    maxlen = max([len(x) for x in insts])
    batch_seq = np.array([x + [Constants.PAD] * (maxlen - len(x)) for x in insts])
    # batch_pos = np.array([[i+1 if w != Constants.PAD else 0 for i, w in enumerate(inst)] for inst in batch_seq])
    batch_pos = np.array([[i if w != Constants.PAD else 0 for i, w in enumerate(inst, 1)] for inst in batch_seq])
    return torch.LongTensor(batch_seq), torch.LongTensor(batch_pos)

def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    #seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = zip(*insts)
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

class Fra2EngDatasets(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
