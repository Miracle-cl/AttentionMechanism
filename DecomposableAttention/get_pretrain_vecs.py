import numpy as np
import h5py
import re
import sys

def load_glove_vec(fname, vocab):
    word_vecs = {}
    with open(fname, 'r') as f:
        for line in f:
            d = line.split(' ')
            word = d[0]
            vec = np.asarray(d[1:], dtype='float32')
            if word in vocab:
                word_vecs[word] = vec

    # for line in open(fname, 'r'):
    #     d = line.split()
    #     word = d[0]
    #     vec = np.array(map(float, d[1:]))
    #
    #     if word in vocab:
    #         word_vecs[word] = vec
    return word_vecs

def main():
    dictionary = './data_snli.word.dict'
    glove_path = '/data/glove.840B.300d.txt'
    outputfile = './data_glove.hdf5'
    vocab = open(dictionary, "r").read().split("\n")[:-1]
    vocab = map(lambda x: (x.split()[0], int(x.split()[1])), vocab) #39079
    word2idx = {x[0]: x[1] for x in vocab}
    print("vocab size is " + str(len(word2idx)))
    w2v_vecs = np.random.normal(size = (len(word2idx), 300))
    w2v = load_glove_vec(glove_path, word2idx)

    print("num words in pretrained model is " + str(len(w2v))) # 38977
    for word, vec in w2v.items():
        w2v_vecs[word2idx[word] - 1] = vec
    for i in range(len(w2v_vecs)):
        w2v_vecs[i] = w2v_vecs[i] / np.linalg.norm(w2v_vecs[i]) # normilize by divide ||v||2
    with h5py.File(outputfile, "w") as f:
        f["word_vecs"] = np.array(w2v_vecs)


if __name__ == '__main__':
    main()
