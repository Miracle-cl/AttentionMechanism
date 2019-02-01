import pickle
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import Dataset

def collate_fn(insts, PAD_token=1):
    # if seq_pad in class then all seqs with same length
    maxlen = max([len(x) for x in insts])
    #maxlen = 24
    seq = np.array([x + [PAD_token] * (maxlen - len(x)) for x in insts])
    seq_lens = np.array([len(x) for x in insts])
    return torch.LongTensor(seq), torch.LongTensor(seq_lens)

def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = zip(*seq_pairs)
    src_insts = collate_fn(src_insts)
    # tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, tgt_insts)

class IMDBdatasets(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
    
class StructuredSelfAttn(nn.Module):
    def __init__(self, input_size, embed_dim, n_layers=1, hidden_size=256, da=350, r=4, n_classes=2, drop_prob=0.5, embed_weights=None):
        super(StructuredSelfAttn, self).__init__()
        self.r = r
        if embed_weights is None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embed_weights), freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, embed_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.attn1 = nn.Linear(2 * hidden_size, da)
        self.attn2 = nn.Linear(da, r)
        self.dropout2 = nn.Dropout(drop_prob)
        # self.fc = nn.Linear(2 * hidden_size * r, n_classes) # flatten
        self.fc = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, input, input_len):
        embeded = self.dropout1(self.embedding(input)) # B x seqlen x embed_dim
        #rnn_output, (_, _) = self.bilstm(embeded) # B x seqlen x 2hs
        packed, seq_len = pack_padded_sequence(embeded, lengths=input_len, batch_first=True)
        (rnn_output, _), (_, _) = self.bilstm(PackedSequence(packed, seq_len))
        rnn_output, _ = pad_packed_sequence(PackedSequence(rnn_output, seq_len), batch_first=True)
        
        # attn_matrix : annotation matrix A in paper
        attn_matrix = torch.tanh(self.attn1(rnn_output)) # B x seqlen x da
        attn_matrix = self.attn2(attn_matrix) # B x seqlen x r
        ## mask : <PAD> == 1
        mask = input.unsqueeze(2).expand_as(attn_matrix) # B x seqlen x r
        penalized_attn_matrix = attn_matrix - 10000.0 * (mask == 1).float() # B x seqlen x r
        penalized_attn_matrix = F.softmax(penalized_attn_matrix.permute(0, 2, 1), dim=2) #  B x r x seqlen  
        
        sentence_embedding = torch.bmm(penalized_attn_matrix, rnn_output) # M in paper : B x r x 2hs
        ## average or flatten
        sent_enc = torch.sum(sentence_embedding, 1) / self.r # average : B x 2hs
        # sent_enc = sentence_embedding.view(sentence_embedding.size(0), -1) # flatten : B x (rx2hs)
        out = self.fc(self.dropout2(sent_enc))
        return out, attn_matrix # penalized_attn_matrix not better

def frobenius_norm(mat):
    # mat : B x r x r
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def train_epoch(model, device, epoch, train_loader, test_loader, criterion, optimizer, 
                clip=0.5, penalization_coeff=2.0, penalization=True):
    # print("There are {} batches in one epoch.".format( len(train_loader) ))
    model.train()

    train_loss = 0
    t0 = time.time()

    for i, batch in enumerate(train_loader, 1):
        src, src_len, tgt = batch
        src = src.to(device)
        tgt = torch.tensor(tgt).to(device) # tgt without modified - cuda out of memory

        optimizer.zero_grad() # here same as model.zero_grad()

        outputs, attn_mat = model(src, src_len)
        # print(outputs.size(), tgt.size())
        loss = criterion(outputs, tgt) # such as tensor(0.6915, device='cuda:1', grad_fn=<NllLossBackward>)
        if penalization:
            # attn_mat : B x r x seqlen
            attn_mat_T = attn_mat.permute(0, 2, 1) # B x seqlen x r
            penalization_loss = frobenius_norm(torch.bmm(attn_mat, attn_mat_T) -  torch.eye(attn_mat.size(1)).to(device))
            loss += penalization_loss * penalization_coeff
        
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
            src, src_len, tgt = batch
            
            total += len(tgt)
            src = src.to(device)
            tgt = torch.tensor(tgt).to(device)
            outputs, _ = model(src, src_len)
            loss = criterion(outputs, tgt)
            eval_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            corr += ( pred.cpu().numpy() == tgt.cpu().numpy() ).sum()

        eval_loss = eval_loss / len(test_loader)
        accuracy = corr / total
        
    return model, train_loss, eval_loss, accuracy

def main(BATCH_SIZE = 500, MAX_LEN = 400):
    with open('../data/imdb_data.pkl', 'rb') as f:
        data_split = pickle.load(f)

    x_train, y_train = data_split['train']['x'], data_split['train']['y']
    x_test, y_test = data_split['test']['x'], data_split['test']['y']
    x_train = [x[:MAX_LEN] for x in x_train]
    x_test = [x[:MAX_LEN] for x in x_test]
    embed_matrix = np.load('imdb_EmbeddingMatrix.npy')
    input_size, embed_dim = embed_matrix.shape

    train_loader = torch.utils.data.DataLoader(
                        IMDBdatasets(x_train, y_train),
                        num_workers = 2,
                        batch_size = BATCH_SIZE,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        IMDBdatasets(x_test, y_test),
                        num_workers = 2,
                        batch_size = BATCH_SIZE,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    struc_selfattn = StructuredSelfAttn(input_size, embed_dim, embed_weights=embed_matrix)
    #sgru = torch.nn.DataParallel(sgru, device_ids=[0, 1], dim=0)
    struc_selfattn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(struc_selfattn.parameters())

    n_epochs = 30
    best_eval_loss = float('inf')

    for epoch in range(1, 1+n_epochs):
        struc_selfattn, train_loss, eval_loss, eval_acc = train_epoch(struc_selfattn, device, epoch, 
                                                                      train_loader, test_loader, criterion, optimizer)

        print(">> Epoch : {} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
              (epoch, train_loss, eval_loss, eval_acc))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(struc_selfattn.state_dict(), 'ssa_imdb_0130.pt')

if __name__ == "__main__":
    main()
    # without penalization the best eval accurcay is 0.88
    # with penalization the best eval accurcay is 0.879