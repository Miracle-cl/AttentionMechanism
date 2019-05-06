''' Translate input text with trained model. '''

import torch
import torch.utils.data
import pickle
from tqdm import tqdm

from model import Transformer
from translator import Translator
from dataset import Fra2EngDatasets, paired_collate_fn
from preprocess import Lang
from train import Args
import Constants

class Translate_params:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.beam_size = 3
        self.batch_size = 128
        self.model = 'best_model.chkpt'
        self.n_best = 1
        self.output = 'pred.txt'

def main():
    '''Main Function'''
    translate_args = Translate_params()

    with open('result_0326.pkl', 'rb') as f:
        data = pickle.load(f)

    eng_lang = data['eng_lang']
    fra_lang = data['fra_lang']
    filter_tokenize = data['token']
    fra_token, eng_token = zip(*filter_tokenize)

    translate_loader = torch.utils.data.DataLoader(
                    Fra2EngDatasets(fra_token[130000:], eng_token[130000:]),
                    num_workers = 2,
                    batch_size = translate_args.batch_size,
                    collate_fn = paired_collate_fn,
                    shuffle = False,
                    drop_last = True)

    translator = Translator(translate_args)

    # for batch in translate_loader:
    #     src_seq, src_pos, tgt_seq, tgt_pos = batch
    #
    #     all_hyp, all_scores = translator.translate_batch(src_seq, src_pos)
    #     print(all_hyp)
    #     break

    with open(translate_args.output, 'w') as f:
        for batch in tqdm(translate_loader, mininterval=2, desc='  - (Translate)', leave=False):
            src_seq, src_pos, tgt_seq, tgt_pos = batch
            all_hyp, all_scores = translator.translate_batch(src_seq, src_pos)

            for i, idx_seqs in enumerate(all_hyp):
                target_line = ' '.join([eng_lang.idx2word[idx.item()] for idx in tgt_seq[i] if idx])
                #print("==: ", target_line)
                f.write("==: " + target_line + '\n')
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([eng_lang.idx2word[idx] for idx in idx_seq])
                    #print("<<: ", pred_line)
                    f.write("<<: " + pred_line + '\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    main()
