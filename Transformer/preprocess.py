# import torch
# import torch.nn as nn
# from torch.utils.data.dataset import Dataset
import unicodedata
import re
import pickle
import numpy as np
import Constants
from collections import Counter

class Lang():
    def __init__(self, sents, name, min_count=30):
        self.name = name
        self.sents = sents
        self.min_count = min_count
        word2idx = {Constants.PAD_WORD: Constants.PAD, Constants.UNK_WORD: Constants.UNK,
                    Constants.BOS_WORD: Constants.BOS, Constants.EOS_WORD: Constants.EOS}
        self.word2idx = {word : idx for word, idx in word2idx.items()}
        self.idx2word = {idx : word for word, idx in word2idx.items()}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for sent in self.sents:
            words += sent.split(' ')

        cc = 4
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count:
                self.word2idx[word] = cc
                self.idx2word[cc] = word
                cc += 1
        return cc


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def data_process_save():
    MIN_LENGTH = Constants.MIN_LENGTH
    # MAX_LENGTH = 25 # avg +- n * std
    BOS_token = Constants.BOS
    EOS_token = Constants.EOS
    UNK_token = Constants.UNK

    with open('../../data/eng-fra.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = []
    for pair in lines:
        eng, fra = pair.split('\t')
        pairs.append([normalize_string(fra), normalize_string(eng)])

    fra_sents = [p[0] for p in pairs]
    eng_sents = [p[1] for p in pairs]

    fra_lang = Lang(fra_sents, 'french')
    eng_lang = Lang(eng_sents, 'english')

    filter_tokenize = []
    for pair in pairs:
        # max_len = 25
        fra1 = [BOS_token] + [fra_lang.word2idx.get(word, UNK_token) for word in pair[0].split(' ')] + [EOS_token]
        eng1 = [BOS_token] + [eng_lang.word2idx.get(word, UNK_token) for word in pair[1].split(' ')] + [EOS_token]
        # if len(eng1) > MIN_LENGTH and len(eng1) < MAX_LENGTH and len(fra1) > MIN_LENGTH and len(fra1) < MAX_LENGTH:
        if len(eng1) > MIN_LENGTH and len(fra1) > MIN_LENGTH:
            filter_tokenize.append([fra1, eng1])
    print(len(filter_tokenize), ' / ', len(pairs))

    # fra_token, eng_token = zip(*filter_tokenize)
    result_dict = {'eng_lang': eng_lang, 'fra_lang': fra_lang, 'token': filter_tokenize}
    with open('result_0326.pkl', 'wb') as pl:
        pickle.dump(result_dict, pl)

def main():
    data_process_save()

if __name__ == '__main__':
    main()
