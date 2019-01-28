import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import unicodedata
import re
import pickle
from collections import Counter
import numpy as np

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

class Lang():
    def __init__(self, sents, name, min_count=30):
        self.name = name
        self.sents = sents
        self.min_count = min_count
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for sent in self.sents:
            words += sent.split(' ')

        cc = 3
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count:
                self.word2idx[word] = cc
                self.idx2word[cc] = word
                cc += 1
        return cc

def main():
    MIN_LENGTH = 2
    # MAX_LENGTH = 25 # avg +- n * std
    EOS_token = 2

    with open('../../data/eng-fra.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = []
    for pair in lines:
        eng, fra = pair.split('\t')
        pairs.append([normalize_string(fra), normalize_string(eng)])

    eng_sents = [p[1] for p in pairs]
    fra_sents = [p[0] for p in pairs]

    eng_lang = Lang(eng_sents, 'english')
    fra_lang = Lang(fra_sents, 'french')

    filter_tokenize = []
    for pair in pairs:
        eng1 = [eng_lang.word2idx[word] for word in pair[1].split(' ') if word in eng_lang.word2idx] + [EOS_token]
        fra1 = [fra_lang.word2idx[word] for word in pair[0].split(' ') if word in fra_lang.word2idx] + [EOS_token]
        # if len(eng1) > MIN_LENGTH and len(eng1) < MAX_LENGTH and len(fra1) > MIN_LENGTH and len(fra1) < MAX_LENGTH:
        if len(eng1) > MIN_LENGTH and len(fra1) > MIN_LENGTH:
            filter_tokenize.append([fra1, eng1])
    print(len(filter_tokenize), ' / ', len(pairs))

    # fra_token, eng_token = zip(*filter_tokenize)
    result_dict = {'eng_lang': eng_lang, 'fra_lang': fra_lang, 'token': filter_tokenize}
    with open('../result.pkl', 'wb') as pl:
        pickle.dump(result_dict, pl)

if __name__ == '__main__':
    main()
