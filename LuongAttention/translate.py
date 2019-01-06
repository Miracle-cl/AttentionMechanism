# translate 1 batch

## encoder / decoder load model state dict
def translate():
    encoder.eval()
    decoder.eval()

    batch_size = 1

    with torch.no_grad():
        input_bs, input_lens, target_bs, target_lens = random_batch(batch_size)
        input_sents = ' '.join([eng_lang.idx2word[i.item()] for i in input_bs.view(-1)])
        target_sents = ' '.join([fra_lang.idx2word[i.item()] for i in target_bs.view(-1)])
        print('>', input_sents)
        print('=', target_sents)

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

        # output_seqs = torch.zeros(max_target_length, batch_size, dtype=torch.int)
        decoded_words = []
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            # decoder_input size - torch.size([5])
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            ) # decoder_output - B x output_size
            all_decoder_outputs[t] = decoder_output # Store this step's outputs
            topv, topi = decoder_output.topk(1)
            # print(topi.squeeze(0))
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append( fra_lang.idx2word[topi.item()] )

            decoder_input = topi.squeeze(0).detach()  # detach from history as input  # topi.view(-1)

        print('<', ' '.join(decoded_words))
        print('')
        return decoded_words

if __name__ == "__main__":
    for _ in range(10):
        translate()
