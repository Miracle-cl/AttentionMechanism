import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderPack(nn.Module):
    # changable length
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(EncoderPack, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True) # n_layers = 1 - dropout = 0
        self.fc = nn.Linear(hidden_size * 2, hidden_size) # encoder hidden size to decoder hidden size

    def forward(self, src, src_lens, hidden=None):
        embedded = self.dropout(self.embed(src))
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_lens)
        # out : length x B x 2h; hiddene : 2l x B x h
        out, hidden = self.gru(packed, hidden)
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(out)
        # hidden [-1, :, :] last backward hidden
        # hidden [-2, :, :] last forward hidden
        hidden = torch.tanh( self.fc( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ) ) # B x h
        return out, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hs, dec_hs):
        # in this function encoder hidden size == decoder hidden size
        super(BahdanauAttention, self).__init__()
        self.attn = nn.Linear(2 * enc_hs + dec_hs, dec_hs)
        self.v = nn.Parameter(torch.randn(dec_hs)) # can use nn.Linear to replace nn.Parameter

    def forward(self, hidden, encoder_outputs):
        # hidden : B x h; encoder_outputs : length x B x 2h
        enc_seqlen = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, enc_seqlen, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # B x l x 2h
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) # B x l x h
        energy = energy.permute(0, 2, 1) # B x h x l
        para = self.v.repeat(batch_size, 1).unsqueeze(1) # h - Bxh - Bx1xh
        attention = torch.bmm(para, energy).squeeze(1) # B x l
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, enc_hs, dec_hs, dropout, bahdanau_attn):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.output_size = output_size
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.attn = bahdanau_attn
        self.gru = nn.GRU(embed_size + 2 * enc_hs, dec_hs)
        self.fc = nn.Linear(embed_size + 2 * enc_hs + dec_hs, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # input : [B]
        # last_hidden : [B, decoder_hidden_size]
        # encoder_outputs : [length, B, 2 * decoder_hidden_size]
        embeded = self.embed(input.unsqueeze(0)) # 1 x B x embed_size
        embeded = self.dropout(embeded)
        # calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden, encoder_outputs) # B x l
        attn_weights = attn_weights.unsqueeze(1) # B x 1 x l
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # B x l x 2h
        weighted_enc_out = torch.bmm(attn_weights, encoder_outputs) # B x 1 x 2h
        weighted_enc_out = weighted_enc_out.permute(1, 0, 2)
        # stack weighted encoder outputs and decoder inputs
        gru_input = torch.cat((embeded, weighted_enc_out), dim=2) # 1 x B x (embed_size+2h)
        gru_out, last_hidden = self.gru(gru_input, last_hidden.unsqueeze(0)) # 1 x B x dec_hs
        assert (gru_out == last_hidden).all()

        # stack embeded, weighted_enc_out, gru_out
        fc_input = torch.cat((embeded.squeeze(0), weighted_enc_out.squeeze(0), gru_out.squeeze(0)), dim=1)
        fc_out = self.fc(fc_input) # B x output_size
        return fc_out, last_hidden.squeeze(0)    

class Seq2SeqPack(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqPack, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, src_lens, teacher_forcing_ratio=0.5):
        # src/tgt : sent_len x B
        # teacher_forcing_ratio is probability to use teacher forcing
        SOS_token = 1
        out_size = self.decoder.output_size
        batch_size = src.size(1)
        max_tgt_len = tgt.size(0)
        dec_outs = torch.zeros(max_tgt_len, batch_size, out_size).to(self.device)
        
        enc_outs, hidden = self.encoder(src, src_lens)

        dec_input = torch.LongTensor([SOS_token] * batch_size).to(self.device)
        for i in range(max_tgt_len):
            dec_out, hidden = self.decoder(dec_input, hidden, enc_outs)
            dec_outs[i] = dec_out
            teacher_force = random.random() < teacher_forcing_ratio
            _, topi = dec_out.max(1)
            dec_input = (tgt[i] if teacher_force else topi)
        return dec_outs


## ============================= Encoder without packing-padding =============================
# class Encoder(nn.Module):
#     def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.embed = nn.Embedding(input_size, embed_size)
#         self.dropout = nn.Dropout(dropout)
#         self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True) # n_layers = 1 - dropout = 0
#         self.fc = nn.Linear(hidden_size * 2, hidden_size) # encoder hidden size to decoder hidden size

#     def forward(self, src, hidden=None):
#         embedded = self.dropout(self.embed(src))
#         # out : length x B x 2h; hiddene : 2l x B x h
#         out, hidden = self.gru(embedded, hidden)
#         # hidden [-1, :, :] last backward hidden
#         # hidden [-2, :, :] last forward hidden
#         hidden = torch.tanh( self.fc( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ) ) # B x h
#         return out, hidden

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device

#     def forward(self, src, tgt, teacher_forcing_ratio=0.5):
#         # src/tgt : sent_len x B
#         # teacher_forcing_ratio is probability to use teacher forcing
#         SOS_token = 1
#         out_size = self.decoder.output_size
#         batch_size = src.size(1)
#         max_tgt_len = tgt.size(0)
#         dec_outs = torch.zeros(max_tgt_len, batch_size, out_size).to(self.device)
        
#         enc_outs, hidden = self.encoder(src)

#         dec_input = torch.LongTensor([SOS_token] * batch_size).to(self.device)
#         for i in range(max_tgt_len):
#             dec_out, hidden = self.decoder(dec_input, hidden, enc_outs)
#             dec_outs[i] = dec_out
#             teacher_force = random.random() < teacher_forcing_ratio
#             _, topi = dec_out.max(1)
#             dec_input = (tgt[i] if teacher_force else topi)
#         return dec_outs


