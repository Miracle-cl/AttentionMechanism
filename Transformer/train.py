import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import math
import time
import pickle
from tqdm import tqdm

from Optim import ScheduledOptim
from model import Transformer
from dataset import Fra2EngDatasets, paired_collate_fn
from preprocess import Lang
import Constants

class Args:
    def __init__(self, src_vocab_size, tgt_vocab_size):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_token_seq_len = 65 # max(65 and 54)
        self.d_word_vec = 512
        self.d_model = 512
        self.d_inner_hid = 2048
        self.n_layers = 6
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
        self.dropout = 0.1
        self.proj_share_weight = True # proj_share_weight : tgt_emb_prj_weight_sharing
        self.embs_share_weight = False # emb_src_tgt_weight_sharing : embs_share_weight

        self.epochs = 2
        self.label_smoothing = True
        self.n_warmup_steps = 4000



def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1] # (b*seq_len, )
    gold = gold.contiguous().view(-1) # (b*seq_len, )
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # (b*seq_len, )

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1) # one-hot embedding of gold ; size (b*seq_len, vocab_size)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) # if == 1 then 0.9 else 0.1 / (vocab_size - 1)
        log_prb = F.log_softmax(pred, dim=1) # (b*seq_len, vocab_size)

        non_pad_mask = gold.ne(Constants.PAD) # if == 0 then 0 else 1
        loss = -(one_hot * log_prb).sum(dim=1) # (b*seq_len, )
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        # example : tgt_seq: b x 16; gold: b x 15; pred: b x 15
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''
    valid_accus = []
    for epoch_i in range(opt.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
                        model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if valid_accu >= max(valid_accus):
            torch.save(checkpoint, 'best_model.chkpt')
            print('    - [Info] The checkpoint file has been updated.')


def main(batchsize = 256, multi_gpu = False):
    with open('result_0326.pkl', 'rb') as f:
        data = pickle.load(f)

    src_vocab_size = data['fra_lang'].n_words
    tgt_vocab_size = data['eng_lang'].n_words
    args = Args(src_vocab_size, tgt_vocab_size)
    eng_lang = data['eng_lang']
    fra_lang = data['fra_lang']
    filter_tokenize = data['token']
    fra_token, eng_token = zip(*filter_tokenize)

    train_loader = torch.utils.data.DataLoader(
                    Fra2EngDatasets(fra_token[:120000], eng_token[:120000]),
                    num_workers = 2,
                    batch_size = batchsize,
                    collate_fn = paired_collate_fn,
                    shuffle = True,
                    drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                    Fra2EngDatasets(fra_token[120000:130000], eng_token[120000:130000]),
                    num_workers = 2,
                    batch_size = batchsize,
                    collate_fn = paired_collate_fn,
                    shuffle = True,
                    drop_last = True)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # print(device)
    transformer = Transformer(
            args.src_vocab_size,
            args.tgt_vocab_size,
            args.max_token_seq_len,
            tgt_emb_prj_weight_sharing = args.proj_share_weight,
            emb_src_tgt_weight_sharing = args.embs_share_weight,
            d_k = args.d_k,
            d_v = args.d_v,
            d_model = args.d_model,
            d_word_vec = args.d_word_vec,
            d_inner = args.d_inner_hid,
            n_layers = args.n_layers,
            n_head = args.n_head,
            dropout = args.dropout)

    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer = nn.DataParallel(transformer, device_ids=[0, 1], dim=0)
        transformer.to(device)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
        transformer.to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(transformer, train_loader, test_loader, optimizer, device, args)

if __name__ == '__main__':
    main(batchsize = 512, multi_gpu = True)
