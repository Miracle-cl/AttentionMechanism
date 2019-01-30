from dataset import Lang, Fra2EngDatasets, paired_collate_fn
from Models import EncoderRNN, LuongAttnDecoderRNN, Seq2Seq
import Constants

import time
import torch
import torch.nn as nn
import numpy as np
import pickle

def train_epoch(model, epoch, train_loader, test_loader, criterion, optimizer, device, clip=5., batch_sz=Constants.batch_size):
    print("There are {} batches in one epoch.".format( len(train_loader) ))
    model.train()
    train_loss = 0
    log_loss_info = []
    t0 = time.time()
    for i, batch in enumerate(train_loader, 1):
        src, src_lens, tgt, tgt_lens = batch
        # calculate max tgt length - batches divided to 2 parts , if not they have the two different seq-lengths
        max_tgt_len = torch.max(tgt_lens).item()
        src = src.to(device)
        tgt = tgt.to(device) # tgt without modified - cuda out of memory
        optimizer.zero_grad() # here same as optimizer.zero_grad()
        outputs = model(src, tgt, src_lens, tgt_lens, max_tgt_len)
        loss = criterion(outputs.view(-1, outputs.size(2)), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        if i % 2 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                                (epoch, i, (time.time()-t0)/60., train_loss/i)
            print(log_str)
            log_loss_info.append(log_str)
            t0 = time.time()
            
    train_loss = train_loss / len(train_loader)
    # print(train_loss)

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src, src_lens, tgt, tgt_lens = batch
            max_tgt_len = torch.max(tgt_lens).item()
            src = src.to(device)
            tgt = tgt.to(device)
            outputs = model(src, tgt, src_lens, tgt_lens, max_tgt_len)
            loss = criterion(outputs.view(-1, outputs.size(2)), tgt.view(-1))
            eval_loss += loss.item()
            # print('over')
        eval_loss = eval_loss / len(test_loader)
        # print(eval_loss)

    return train_loss, eval_loss, log_loss_info

def main(multi_gpu = False):
    with open('result.pkl', 'rb') as pl:
        rd = pickle.load(pl)

    filter_tokenize = rd['token']
    fra_lang = rd['fra_lang']
    eng_lang = rd['eng_lang']
    fra_token, eng_token = zip(*filter_tokenize)

    train_loader = torch.utils.data.DataLoader(
                        Fra2EngDatasets(fra_token[:130000], eng_token[:130000]),
                        num_workers = 2,
                        batch_size = Constants.batch_size,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        Fra2EngDatasets(fra_token[130000:], eng_token[130000:]),
                        num_workers = 2,
                        batch_size = Constants.batch_size,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    INPUT_DIM = fra_lang.n_words
    OUTPUT_DIM = eng_lang.n_words
    
    ENC_EMBED_SIZE = 300
    DEC_EMBED_SIZE = 300
    
    HIDDEN_SIZE = 256

    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    fra_embed_matrix = np.load('Fra_EmbeddingMatrix.npy')
    eng_embed_matrix = np.load('Eng_EmbeddingMatrix.npy')
    
    encoder = EncoderRNN(INPUT_DIM, ENC_EMBED_SIZE, HIDDEN_SIZE, weights=fra_embed_matrix, n_layers=2, dropout=ENC_DROPOUT)
    # decoder = LuongAttnDecoderRNN('general', DEC_EMBED_SIZE, HIDDEN_SIZE, OUTPUT_DIM, device, 
    #                               weights=eng_embed_matrix, n_layers=1, dropout=DEC_DROPOUT)
    
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = LuongAttnDecoderRNN('general', DEC_EMBED_SIZE, HIDDEN_SIZE, OUTPUT_DIM, device, 
                                      weights=eng_embed_matrix, n_layers=1, dropout=DEC_DROPOUT)
        s2s_model = Seq2Seq(encoder, decoder, device)
        s2s_model = nn.DataParallel(s2s_model, device_ids=[0, 1], dim=0)
        s2s_model.to(device)
    else:
        device = torch.device("cuda:1")
        decoder = LuongAttnDecoderRNN('general', DEC_EMBED_SIZE, HIDDEN_SIZE, OUTPUT_DIM, device, 
                                      weights=eng_embed_matrix, n_layers=1, dropout=DEC_DROPOUT)
        s2s_model = Seq2Seq(encoder, decoder, device)
        s2s_model.to(device)
        
    optimizer = torch.optim.Adam(s2s_model.parameters())
    # nn.CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
    # criterion = nn.CrossEntropyLoss(ignore_index=0) # PAD_token = 0
    criterion = nn.NLLLoss(ignore_index=0)

    n_epochs = Constants.n_epochs
    best_eval_loss = float('inf')
    MODEL_SAVE_PATH = 's2s_luong_attn_0127.pt'
    log_info = []
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        trainloss, evalloss, log_loss_info = train_epoch(s2s_model, epoch, train_loader, test_loader, criterion, optimizer, device)
        used_time = time.time() - start
        log_str = ">> Epoch : {} , Time : {:.2f} , TrainLoss : {:.4f} , EvalLoss : {:.4f}\n\n".format(epoch, used_time/60., trainloss, evalloss)
        log_loss_info.append(log_str)
        log_info += log_loss_info
        print(log_str)
        if evalloss < best_eval_loss:
            best_eval_loss = evalloss
            torch.save(s2s_model.state_dict(), MODEL_SAVE_PATH)

    with open('log_info_0127.txt', 'w') as wrf:
        for item in log_info:
            wrf.write(item)
            wrf.write('\n')

if __name__ == "__main__":
    main(multi_gpu = True)
