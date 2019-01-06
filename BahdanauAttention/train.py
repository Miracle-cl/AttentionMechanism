import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *
from Models import *


def train_epoch_pack(model, epoch, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader, 1):
        src, src_lens, tgt, tgt_lens = batch
        src = src.permute(1, 0).to(device)
        tgt = tgt.permute(1, 0).to(device) # tgt without modified - cuda out of memory
        optimizer.zero_grad() # here same as optimizer.zero_grad()
        outputs = model(src, tgt, src_lens)
        loss = criterion(outputs.view(-1, outputs.size(2)), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    # print(train_loss)

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src, src_lens, tgt, tgt_lens = batch
            src = src.permute(1, 0).to(device)
            tgt = tgt.permute(1, 0).to(device)
            outputs = model(src, tgt, src_lens)
            loss = criterion(outputs.view(-1, outputs.size(2)), tgt.view(-1))
            eval_loss += loss.item()
        eval_loss = eval_loss / len(test_loader)
        # print(eval_loss)

    return train_loss, eval_loss

def main():
    with open('result.pkl', 'rb') as pl:
        rd = pickle.load(pl)

    filter_tokenize = rd['token']
    fra_lang = rd['fra_lang']
    eng_lang = rd['eng_lang']

    INPUT_DIM = fra_lang.n_words
    OUTPUT_DIM = eng_lang.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    fra_token, eng_token = zip(*filter_tokenize)

    train_loader = torch.utils.data.DataLoader(
                        Fra2EngDatasets(fra_token[:130000], eng_token[:130000]),
                        num_workers = 2,
                        batch_size = 16,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        Fra2EngDatasets(fra_token[130000:], eng_token[130000:]),
                        num_workers = 2,
                        batch_size = 16,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    device = torch.device('cuda:0')
    ba_attn = BahdanauAttention(ENC_HID_DIM, DEC_HID_DIM)
    enc = EncoderPack(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, 1, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, ba_attn)

    model = Seq2SeqPack(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_epochs = 2
    best_eval_loss = float('inf')
    MODEL_SAVE_PATH = 's2s_bahdanau_attn.pt'
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        trainloss, evalloss = train_epoch_pack(model, epoch, train_loader, test_loader, criterion, optimizer, device)
        used_time = time.time() - start
        print("Epoch : {} , Time : {:.2f} , TrainLoss : {:.4f} , EvalLoss : {:.4f}".format(epoch, used_time, trainloss, evalloss))
        if evalloss < best_eval_loss:
            best_eval_loss = evalloss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
