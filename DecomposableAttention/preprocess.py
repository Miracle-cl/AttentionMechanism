import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["<blank>","<unk>","<s>","</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1

    def convert(self, w):
        # return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(1,100)) + '>']
        return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(1,101)) + '>']

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.items()]
        items.sort()
        for v, k in items:
            # print >>out, k, v # now k, v saved in outfile in python 2
            print(k, v, file=out)

        out.close()

    def prune_vocab(self, k, cnt=False):
        vocab_list = [(word, count) for word, count in self.vocab.items()]
        if cnt:
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list if pair[1] > k}
        else:
            vocab_list.sort(key = lambda x: x[1], reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.strip().split()
            self.d[v] = int(k)

def pad(ls, length, symbol, pad_back = True):
    if len(ls) >= length:
        return ls[:length]
    if pad_back:
        return ls + [symbol] * (length -len(ls))
    else:
        return [symbol] * (length -len(ls)) + ls

def get_glove_words(f):
    glove_words = set()
    for line in open(f, "r"):
        word = line.split()[0].strip()
        glove_words.add(word)
    return glove_words

def get_data(args):
    word_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    label_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    label_indexer.d = {}
    glove_vocab = get_glove_words(args.glove)
    for i in range(1,101): #hash oov words to one of 100 random embeddings, per Parikh et al. 2016
        oov_word = '<oov'+ str(i) + '>'
        word_indexer.vocab[oov_word] += 1
    def make_vocab(srcfile, targetfile, labelfile, seqlength):
        num_sents = 0
        for _, (src_orig, targ_orig, label_orig) in \
                enumerate(zip(open(srcfile,'r'), open(targetfile,'r'), open(labelfile, 'r'))):
            src_orig = word_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            label = label_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                continue
            num_sents += 1
            for word in targ:
                if word in glove_vocab:
                    word_indexer.vocab[word] += 1

            for word in src:
                if word in glove_vocab:
                    word_indexer.vocab[word] += 1

            for word in label:
                label_indexer.vocab[word] += 1

        return num_sents

    def convert(srcfile, targetfile, labelfile, batchsize, seqlength, outfile, num_sents, max_sent_l=0, shuffle=0):
        newseqlength = seqlength + 1 #add 1 for BOS
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        labels = np.zeros((num_sents,), dtype =int)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        both_lengths = np.zeros(num_sents, dtype = {'names': ['x','y'], 'formats': ['i4', 'i4']})
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig, label_orig) in \
                enumerate(zip(open(srcfile,'r'), open(targetfile,'r') ,open(labelfile,'r'))):
            src_orig = word_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())
            targ =  [word_indexer.BOS] + targ_orig.strip().split()
            src =  [word_indexer.BOS] + src_orig.strip().split()
            label = label_orig.strip().split()
            max_sent_l = max(len(targ), len(src), max_sent_l)
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 2 or len(src) < 2:
                dropped += 1
                continue
            targ = pad(targ, newseqlength, word_indexer.PAD)
            targ = word_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)

            src = pad(src, newseqlength, word_indexer.PAD)
            src = word_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)

            targets[sent_id] = np.array(targ,dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()
            labels[sent_id] = label_indexer.d[label[0]]
            both_lengths[sent_id] = (source_lengths[sent_id], target_lengths[sent_id])
            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

        print(sent_id, num_sents)
        if shuffle == 1:
            rand_idx = np.random.permutation(sent_id)
            targets = targets[rand_idx]
            sources = sources[rand_idx]
            source_lengths = source_lengths[rand_idx]
            target_lengths = target_lengths[rand_idx]
            labels = labels[rand_idx]
            both_lengths = both_lengths[rand_idx]

        #break up batches based on source/target lengths


        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths)

        both_lengths = both_lengths[:sent_id]
        sorted_lengths = np.argsort(both_lengths, order = ('x', 'y'))
        sources = sources[sorted_lengths]
        targets = targets[sorted_lengths]
        labels = labels[sorted_lengths]
        target_l = target_lengths[sorted_lengths]
        source_l = source_lengths[sorted_lengths]

        curr_l_src = 0
        curr_l_targ = 0
        l_location = [] #idx where sent length changes

        for j,i in enumerate(sorted_lengths):
            if source_lengths[i] > curr_l_src or target_lengths[i] > curr_l_targ:
                curr_l_src = source_lengths[i]
                curr_l_targ = target_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        batch_l = []
        target_l_new = []
        source_l_new = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])
            source_l_new.append(source_l[batch_idx[i]-1])
            target_l_new.append(target_l[batch_idx[i]-1])
        # Write output
        f = h5py.File(outfile, "w")
        f["source"] = sources
        f["target"] = targets
        f["target_l"] = np.array(target_l_new, dtype=int)
        f["source_l"] = np.array(source_l_new, dtype=int)
        f["label"] = np.array(labels, dtype=int)
        f["label_size"] = np.array([len(np.unique(np.array(labels, dtype=int)))])
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["source_size"] = np.array([len(word_indexer.d)])
        f["target_size"] = np.array([len(word_indexer.d)])
        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()
        return max_sent_l

    print("First pass through data to get vocab...")

    num_sents_train = make_vocab(args.srcfile, args.targetfile, args.labelfile,
                                             args.seqlength)
    print("Number of sentences in training: {}".format(num_sents_train))
    num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile, args.labelvalfile,
                                             args.seqlength)
    print("Number of sentences in valid: {}".format(num_sents_valid))
    num_sents_test = make_vocab(args.srctestfile, args.targettestfile, args.labeltestfile,
                                             args.seqlength)
    print("Number of sentences in test: {}".format(num_sents_test))

    #prune and write vocab
    word_indexer.prune_vocab(0, True)
    label_indexer.prune_vocab(1000)
    if args.vocabfile != '':
        print('Loading pre-specified source vocab from ' + args.vocabfile)
        word_indexer.load_vocab(args.vocabfile)
    word_indexer.write(args.outputfile + ".word.dict")
    label_indexer.write(args.outputfile + ".label.dict")
    print("Source vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab),
                                                          len(word_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab),
                                                          len(word_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.labelvalfile,
                         args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_sent_l, args.shuffle)
    max_sent_l = convert(args.srcfile, args.targetfile, args.labelfile,
                         args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train,
                         max_sent_l, args.shuffle)
    max_sent_l = convert(args.srctestfile, args.targettestfile, args.labeltestfile,
                         args.batchsize, args.seqlength,
                         args.outputfile + "-test.hdf5", num_sents_test,
                         max_sent_l, args.shuffle)
    print("Max sent length (before dropping): {}".format(max_sent_l))


class Args():
    def __init__(self):
        out_folder = './data_snli/'
        self.srcfile = out_folder + "src-train.txt"
        self.srcvalfile = out_folder + "src-dev.txt"
        self.srctestfile = out_folder + "src-test.txt"

        self.targetfile = out_folder + "targ-train.txt"
        self.targetvalfile = out_folder + "targ-dev.txt"
        self.targettestfile = out_folder + "targ-test.txt"

        self.labelfile = out_folder + "label-train.txt"
        self.labelvalfile = out_folder + "label-dev.txt"
        self.labeltestfile = out_folder + "label-test.txt"

        self.batchsize = 32
        self.seqlength = 1000
        self.outputfile = './data_snli'
        self.vocabfile = ''
        self.shuffle = 1
        self.glove = '/data/glove.840B.300d.txt'

if __name__ == '__main__':
    args = Args()
    get_data(args)
