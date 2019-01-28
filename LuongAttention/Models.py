# modified based on https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import Constants

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, weights=None, n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # self.embedding = nn.Embedding(input_size, embed_size)
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        total_length = embedded.size(1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed)
        # unpack (back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)
        # Sum bidirectional outputs : enc_seq_len x B x hs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # outputs : B x enc_seq_len x hs; hidden : n_layers * 2 x B x hs (backward-forward-b-f)
        # choose the last 2 hidden layers - B x hs
        hidden = torch.tanh( self.fc( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ) )
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden : decoder rnn output
        # (hidden.size() - 1xBxhs, encoder_outputs.size() - enc_seq_lenxBxhs) [layer = 1]
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len).to(self.device) # B x Seq_len

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[b, i].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x Seq_len
        return F.softmax(attn_energies, dim = 1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = (hidden.squeeze()).dot(encoder_output.squeeze())
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = (hidden.squeeze()).dot(energy.squeeze())
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = (self.v.squeeze()).dot(energy.squeeze())
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embed_size, hidden_size, output_size, device, weights=None, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        # Define layers
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, device)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        # input_seq : torch.size([B])
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq) # B x hidden_size
        embedded = self.embedding_dropout(embedded) # B x 1 x hidden_size
        embedded = embedded.view(batch_size, 1, self.embed_size) # S = B x 1 x hidden_size

        # Get current hidden state from input word and last hidden state
        self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embedded, last_hidden) # rnn_output - B x 1 x hidden_size
        # assert (rnn_output == hidden).all() # n_layer == 1

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs) # B x 1 x hidden_size

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1) # S=B x 1 x hidden_size -> B x hidden_size
        context = context.squeeze(1)       # B x 1 x hidden_size -> B x hidden_size
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input)) # B x hidden_size

        # Finally predict next token (Luong eq. 6, without softmax)
        output = F.log_softmax(self.out(concat_output), dim=1) # B x output_size

        # Return final output, hidden state, and attention weights (for visualization)
        # return output, hidden, attn_weights
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_var, tgt_var, src_len, tgt_len, max_tgt_len, teacher_forcing_ratio=0.5):
        SOS_token = Constants.SOS_token
        batch_size = src_var.size(0)
        # if max_tgt_len calculated in model then will be splited, then two max_tgt_len in different GPU will affect
        # max_tgt_len = max(tgt_len)
        # print(max_tgt_len)
        dec_input = torch.LongTensor([SOS_token] * batch_size).to(self.device)
        dec_outputs = torch.zeros(batch_size, max_tgt_len, self.decoder.output_size).to(self.device)

        enc_outputs, hidden = self.encoder(src_var, src_len)
        hidden = hidden.unsqueeze(0) # 1 x B x hs
        for i in range(max_tgt_len):
            dec_out, hidden = self.decoder(dec_input, hidden, enc_outputs)
            dec_outputs[:, i, :] = dec_out
            teacher_force = random.random() < teacher_forcing_ratio
            _, topi = dec_out.max(1)
            dec_input = (tgt_var[:, i] if teacher_force else topi)
        return dec_outputs
