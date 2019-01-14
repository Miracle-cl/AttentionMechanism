from dataset import Lang, Fra2EngDatasets, paired_collate_fn
from Models import EncoderRNN, LuongAttnDecoderRNN, Seq2Seq
import Constants

import time
import torch
import torch.nn as nn
import numpy as np
import pickle

class BeamState:
    def __init__(self, score, sents, hidden):
        self.score = score # probs of sent list
        self.sents = sents # word index list
        self.hidden = hidden

def load_data(data_paths):
    data_pkl, src_matrix, tgt_matrix = data_paths['data_pkl'], data_paths['src_matrix'], data_paths['tgt_matrix']
    with open(data_pkl, 'rb') as pl:
        rd = pickle.load(pl)

    filter_tokenize = rd['token']
    fra_lang = rd['fra_lang']
    eng_lang = rd['eng_lang']
    fra_token, eng_token = zip(*filter_tokenize)
    
    fra_embed_matrix = np.load(src_matrix)
    eng_embed_matrix = np.load(tgt_matrix)

    # translate_loader = torch.utils.data.DataLoader(
    #                         Fra2EngDatasets(fra_token[100000:100030], eng_token[100000:100030]),
    #                         num_workers = 1,
    #                         batch_size = 1,
    #                         collate_fn = paired_collate_fn,
    #                         shuffle = True,
    #                         drop_last = True) # train data

    translate_loader = torch.utils.data.DataLoader(
                            Fra2EngDatasets(fra_token[130000:130030], eng_token[130000:130030]),
                            num_workers = 1,
                            batch_size = 1,
                            collate_fn = paired_collate_fn,
                            shuffle = True,
                            drop_last = True) # test data
    
    return fra_lang, eng_lang, fra_embed_matrix, eng_embed_matrix, translate_loader


def translate(s2s_model, translate_loader, fra_lang, eng_lang):
    # greedy algorithms for translate
    s2s_model.eval()

    batch_size = 1
    SOS_token = Constants.SOS_token
    EOS_token = Constants.EOS_token
    max_tgt_len = 30
    with torch.no_grad():
        for i, batch in enumerate(translate_loader, 1):
            src_var, src_len, tgt_var, tgt_len = batch
            src_var = src_var.permute(1, 0).to(s2s_model.device)
            input_sents = ' '.join([fra_lang.idx2word[i.item()] for i in src_var.view(-1)])
            target_sents = ' '.join([eng_lang.idx2word[i.item()] for i in tgt_var.view(-1)])
            print('>', input_sents)
            print('=', target_sents)

            # outputs = s2s_model(src_var, tgt_var, src_len, tgt_len)
            enc_outputs, hidden = s2s_model.encoder(src_var, src_len)
            hidden = hidden.unsqueeze(0) # 1 x B x hs
            dec_input = torch.LongTensor([SOS_token] * batch_size).to(s2s_model.device)
            dec_words = []
            for i in range(max_tgt_len):
                dec_out, hidden = s2s_model.decoder(dec_input, hidden, enc_outputs)

                _, topi = dec_out.topk(1)
                dec_words.append(eng_lang.idx2word[topi.item()])
                if topi.item() == EOS_token:
                    break
                dec_input = topi

            print('<', ' '.join(dec_words))
            print()

            
def translate_beam(s2s_model, translate_loader, fra_lang, eng_lang, beam_size=2):
    # beam search for translate
    s2s_model.eval()
    
    batch_size = 1
    SOS_token = Constants.SOS_token
    EOS_token = Constants.EOS_token
    max_tgt_len = 30

    with torch.no_grad():
        for j, batch in enumerate(translate_loader, 1):
            src_var, src_len, tgt_var, tgt_len = batch
            src_var = src_var.permute(1, 0).to(s2s_model.device)
            input_sents = ' '.join([fra_lang.idx2word[i.item()] for i in src_var.view(-1)])
            target_sents = ' '.join([eng_lang.idx2word[i.item()] for i in tgt_var.view(-1)])
            print('>', input_sents)
            print('=', target_sents)

            # outputs = s2s_model(src_var, tgt_var, src_len, tgt_len)
            enc_outputs, hidden = s2s_model.encoder(src_var, src_len)
            hidden = hidden.unsqueeze(0) # 1 x B x hs
            dec_input = torch.LongTensor([SOS_token] * batch_size).to(s2s_model.device)

            res = []
            for step in range(max_tgt_len):
                flag = 0
                if step == 0:
                    dec_out, hidden = s2s_model.decoder(dec_input, hidden, enc_outputs)

                    topv, topi = dec_out.topk(beam_size)
                    for i in range(beam_size):
                        res.append(BeamState(topv[0, i].item(), [topi[0, i].item()], hidden))
                else:
                    prev_states = res[:beam_size]
                    next_states = []
                    for bstate in prev_states:
                        if bstate.sents[-1] == EOS_token: # EOS_token == 2
                            next_states.append(bstate)
                            flag += 1
                            continue

                        dec_input = torch.LongTensor([bstate.sents[-1]] * batch_size).to(s2s_model.device)
                        dec_out, hidden = s2s_model.decoder(dec_input, bstate.hidden, enc_outputs)
                        topv, topi = dec_out.topk(beam_size)
                        for i in range(beam_size):
                            new_score = (bstate.score * len(bstate.sents) + topv[0, i].item()) / (len(bstate.sents) + 1) # log_softmax
                            new_sents = bstate.sents + [topi[0, i].item()]
                            next_states.append( BeamState(new_score, new_sents, hidden) )
                    res = sorted(next_states, key=lambda x: x.score, reverse=True)
                    # print(j, step,  [x.score for x in res])
                if flag == beam_size:
                    break
            output_sent = ' '.join([eng_lang.idx2word[i] for i in res[0].sents])
            print('<', output_sent)
            print()

            
def main(data_paths, model_path):
    fra_lang, eng_lang, fra_embed_matrix, eng_embed_matrix, translate_loader = load_data(data_paths)
    
    # model config and load model
    INPUT_DIM = fra_lang.n_words
    OUTPUT_DIM = eng_lang.n_words

    ENC_EMBED_SIZE = 300
    DEC_EMBED_SIZE = 300

    HIDDEN_SIZE = 256

    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(INPUT_DIM, ENC_EMBED_SIZE, HIDDEN_SIZE, weights=fra_embed_matrix, n_layers=2, dropout=ENC_DROPOUT)
    decoder = LuongAttnDecoderRNN('general', DEC_EMBED_SIZE, HIDDEN_SIZE, OUTPUT_DIM, device, weights=eng_embed_matrix, n_layers=1, dropout=DEC_DROPOUT)
    s2s_model = Seq2Seq(encoder, decoder, device)


    s2s_model.load_state_dict(torch.load(model_path))
    s2s_model.to(s2s_model.device)
    
    # translate
    translate(s2s_model, translate_loader, fra_lang, eng_lang)
    # beam_size = 2
    translate_beam(s2s_model, translate_loader, fra_lang, eng_lang)
    
if __name__ == "__main__":
    data_paths = {'data_pkl': 'result.pkl',
                  'src_matrix': 'Fra_EmbeddingMatrix.npy', 
                  'tgt_matrix': 'Eng_EmbeddingMatrix.npy'}
    model_path = 's2s_luong_attn_0113.pt'
    main(data_paths, model_path)




