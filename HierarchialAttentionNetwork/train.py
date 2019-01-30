import pickle
import numpy as np
import time
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_HAN import HierarchialAttentionNetwork

class Lang():
    def __init__(self, sents, stoplist, min_count=30):
        self.sents = sents
        self.stoplist = stoplist
        self.min_count = min_count
        self.word2idx = {"<PAD>": 0}
        self.idx2word = {0: "<PAD>"}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for sent in self.sents:
            words += sent.split(' ')

        cc = 1
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count and word not in self.stoplist:
                self.word2idx[word] = cc
                self.idx2word[cc] = word
                cc += 1
        return cc

class NewsPaperDatasets(Dataset):
    def __init__(self, src, src_sents, src_words, tgt):
        self.src = src
        self.src_sents = src_sents
        self.src_words = src_words
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.src_sents[idx], self.src_words[idx], self.tgt[idx]


def train_epoch(model, device, epoch, train_loader, test_loader, criterion, optimizer, clip=5.):
    # print("There are {} batches in one epoch.".format( len(train_loader) ))
    model.train()

    train_loss = 0
    t0 = time.time()

    for i, batch in enumerate(train_loader, 1):
        src, src_sents, src_words, tgt = [sub_tensor.long().to(device) for sub_tensor in batch]

        optimizer.zero_grad() # here same as model.zero_grad()
        # hidden = hidden.detach()
        outputs, _, _ = model(src, src_sents, src_words)
        # print(outputs.size(), tgt.size())
        loss = criterion(outputs, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        if i % 10 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                        (epoch, i, time.time()-t0, train_loss/i)
            print(log_str)
            t0 = time.time()

    train_loss = train_loss / len(train_loader)
    # print(train_loss)

    model.eval()
    eval_loss = 0

    corr = total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src, src_sents, src_words, tgt = [sub_tensor.long().to(device) for sub_tensor in batch]
            total += tgt.size(0)

            outputs, _, _ = model(src, src_sents, src_words)
            loss = criterion(outputs, tgt)
            eval_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            corr += (pred.cpu().numpy() == tgt.cpu().numpy()).sum()

        eval_loss = eval_loss / len(test_loader)

        accuracy = corr / total
    return model, train_loss, eval_loss, accuracy

def main(batch_size = 256, multi_gpu = False):
    with open('newspaper_han.pkl', 'rb') as f:
        han_data = pickle.load(f)

    x_train, sents_doc_train, words_sent_train, y_train = \
                han_data['trian'][1], han_data['trian'][2], han_data['trian'][3], han_data['trian'][4] # spelling mistakes
    x_test, sents_doc_test, words_sent_test, y_test = \
                han_data['test'][1], han_data['test'][2], han_data['test'][3], han_data['test'][4]

    embed_matrix = np.load('EmbeddingMatrix_han.npy')
    vocab_size = embed_matrix.shape[0]

    train_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_train, sents_doc_train, words_sent_train, y_train),
                        num_workers = 2,
                        batch_size = batch_size,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_test, sents_doc_test, words_sent_test, y_test),
                        num_workers = 2,
                        batch_size = batch_size,
                        shuffle = True,
                        drop_last = True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    han = HierarchialAttentionNetwork(n_classes = 20,
                                        vocab_size = vocab_size,
                                        emb_size = 300,
                                        word_rnn_size = 128,
                                        sentence_rnn_size = 128,
                                        word_rnn_layers = 2,
                                        sentence_rnn_layers = 2,
                                        word_att_size = 128,
                                        sentence_att_size = 128,
                                        dropout = 0.5,
                                        embed_weights = embed_matrix)

    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        han = torch.nn.DataParallel(han, device_ids=[0, 1], dim=0)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    han.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(han.parameters())

    n_epochs = 15
    best_eval_loss = float('inf') # best eval accuracy 0.88

    for epoch in range(1, 1+n_epochs):
        han, train_loss, eval_loss, eval_acc = train_epoch(han, device, epoch, train_loader, test_loader, criterion, optimizer)

        print(">> Epoch : {} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
              (epoch, train_loss, eval_loss, eval_acc))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(han.state_dict(), 'han_newspaper_0127.pt')

if __name__ == "__main__":
    main(batch_size = 256, multi_gpu = False)