import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models import EncoderRNN, LuongAttnDecoderRNN
from masked_crossentropy_loss import *
import random

PAD_token = 0
USE_CUDA = True

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, filter_tokenize):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pa = random.choice(filter_tokenize)
        input_seqs.append(pa[0])
        target_seqs.append(pa[1])

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    target_var = torch.LongTensor(target_padded).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths

def train_batch(input_bs, input_lens, target_bs, target_lens, encoder, decoder,
                encoder_optimizer, decoder_optimizer, clip=5.0):
    encoder.train()
    decoder.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_bs, input_lens)

    # Prepare decoder input and outputs
    batch_size = input_bs.size(1)
    max_target_length = max(target_lens)
    decoder_input = torch.LongTensor([SOS_token] * batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        ) # decoder_output - B x output_size
        all_decoder_outputs[t] = decoder_output # Store this step's outputs
        decoder_input = target_bs[t] # Next input is current target

    # calculate loss
    # Test masked cross entropy loss
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_bs.transpose(0, 1).contiguous(),
        target_lens
    )

    loss.backward()

    # Clip gradient norms
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def main():
    # Configure models
    attn_model = 'dot'
    hidden_size = 500
    n_layers = 2
    dropout = 0.1
    batch_size = 100

    # Configure training/optimization
    clip = 5.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = 10000
    epoch = 0

    with open('result.pkl', 'rb') as pl:
        rd = pickle.load(pl)

    filter_tokenize = rd['token']
    fra_lang = rd['fra_lang']
    eng_lang = rd['eng_lang']

    # Initialize models
    encoder = EncoderRNN(eng_lang.n_words, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, fra_lang.n_words, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    # criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # input_bs, input_lens, target_bs, target_lens = random_batch(batch_size, filter_tokenize)
    # loss = train_batch(input_bs, input_lens, target_bs, target_lens, encoder, decoder,
    #                    encoder_optimizer, decoder_optimizer, clip=5.0)
    n_epochs = 1000
    losses = []

    start = time.time()
    for epoch in range(1, 1+n_epochs):
        input_bs, input_lens, target_bs, target_lens = random_batch(batch_size, filter_tokenize)
        loss = train_batch(input_bs, input_lens, target_bs, target_lens, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, clip=5.0)
        losses.append(loss)
        if epoch % 10 == 0:
            used_time = time.time() - start
            print("Epoch : {} , Time : {} , Loss : {}".format(epoch, used_time, loss))
            start = time.time()

if __name__ == "__main__":
    main()
