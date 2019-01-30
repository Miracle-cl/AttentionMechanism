from sklearn.datasets import fetch_20newsgroups
from nltk import tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import os
import unicodedata
import re
import pickle

class Lang():
    def __init__(self, docs, stoplist, min_count=30):
        self.docs = docs
        self.stoplist = stoplist
        self.min_count = min_count
        self.word2idx = {"<PAD>": 0}
        self.idx2word = {0: "<PAD>"}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for doc in self.docs:
            for sent in doc:
                words += sent.split(' ')

        cc = 1
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count and word not in self.stoplist:
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
    # s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z0-9]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# drop  \, ", '
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

def save_embed_matrix(input_lang):
    ## load pre-trained word2vec
    embed_dim = 300
    with open('/data/charley/crawl-300d-2M.pkl', 'rb') as pf:
        fb_w2v = pickle.load(pf)
        
    embed_matrix = np.zeros((input_lang.n_words, embed_dim))
    flag = 0
    for word, idx in input_lang.word2idx.items():
        word_vec = fb_w2v.get(word)
        if word_vec is not None:
            embed_matrix[idx] = word_vec
            flag += 1
    print("There are {} words with pre-trained vector in vocab_size = {}.".format(flag, len(input_lang.word2idx)))
    np.save('EmbeddingMatrix_han', embed_matrix) # saved as EmbeddingMatrix.npy


def main():
    ## get data 
    stoplist = stopwords.words('english')
    features_all = fetch_20newsgroups(subset='all').data
    targets_all = fetch_20newsgroups(subset='all').target

    # split sentences by nltk
    reviews = []
    for text in features_all:
        text = clean_str(text.encode('ascii','ignore').decode("utf-8"))
        sentences = tokenize.sent_tokenize(text) # list of string senteces
        reviews.append(sentences)

    reviews_clean = [[normalize_string(sent) for sent in doc if normalize_string(sent)] for doc in reviews]

    input_lang = Lang(reviews_clean, stoplist)

    assert input_lang.n_words == len(input_lang.word2idx)

    # word to index
    input_docs = []
    for doc in reviews_clean:
        doc_idx = [[input_lang.word2idx[word] for word in sent.split(' ') if word in input_lang.word2idx] 
                   for sent in doc]
        input_docs.append(doc_idx)
    print("num of docs : ", len(input_docs)) # num of docs
    print("num of sentences in doc-1 : ", len(input_docs[0])) # num of sentences in doc-1
    print("num of words in sentence-1 of doc-1 : ", len(input_docs[0][0])) # num of words in sentence-1 of doc-1

    # drop sentencs without word 
    input_docs_new = []
    for doc in input_docs:
        input_docs_new.append([sent for sent in doc if sent])

    num_docs = len(input_docs_new) # 18846
    ## calculate max_sents, max_sent_length by mean and std
    max_sents = 75 # 15 + 30x2
    max_sent_length = 60
    
    # prepare data for model
    data = np.zeros((num_docs, max_sents, max_sent_length), dtype='int32')
    sentences_per_document = np.zeros(num_docs, dtype='int32') # num of sentences of a doc
    words_per_sentence = np.zeros((num_docs, max_sents), dtype='int32') # num of words of sentences of docs

    for i, doc in enumerate(input_docs_new):
        doc_len = min(len(doc), max_sents)
        doc = doc[:doc_len]
        sentences_per_document[i] = doc_len
        for j, sent in enumerate(doc):
            sent_len = min(len(sent), max_sent_length)
            data[i, j, :sent_len] = sent[:sent_len]
            words_per_sentence[i, j] = sent_len

    ## split data
    x_train, sents_doc_train, words_sent_train, y_train = \
                data[:11314], sentences_per_document[:11314], words_per_sentence[:11314], targets_all[:11314]
    x_test, sents_doc_test, words_sent_test, y_test = \
                data[11314:], sentences_per_document[11314:], words_per_sentence[11314:], targets_all[11314:]
    print(x_train.shape, sents_doc_train.shape, words_sent_train.shape, y_train.shape)


    newspaper_han_data = {'trian': {1:x_train, 2:sents_doc_train, 3:words_sent_train, 4:y_train}, 
                          'test': {1:x_test, 2:sents_doc_test, 3:words_sent_test, 4:y_test},
                          'lang': input_lang}

    # save data
    save_embed_matrix(input_lang)
    
    with open('newspaper_han.pkl', 'wb') as f:
        pickle.dump(newspaper_han_data, f)
        
if __name__ == "__main__":
    main()