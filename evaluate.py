from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import MAX_LENGTH, SOS_token, EOS_token, device

f_read = open('input_lang_word2index.pkl', 'rb')
input_lang_word2index = pickle.load(f_read)
f_read.close()

f_read = open('output_lang_index2word.pkl', 'rb')
output_lang_index2word = pickle.load(f_read)
f_read.close()

def get_pairs(src_path, tgt_path):
    src_lines = open(src_path, encoding='utf-8').read().strip().split('\n')
    tgt_lines = open(tgt_path, encoding='utf-8').read().strip().split('\n')

    pairs = []
    for i in range(len(src_lines)):
        lines = [src_lines[i], tgt_lines[i]]
        pairs.append(lines)
    return pairs


# def indexesFromSentence(sentence):
#     indexes = []
#     for word in sentence.split(' '):
#         if word in input_lang_word2index:
#             index = input_lang_word2index[word]
#             indexes.append(index)
#         else:
#             indexes.append(3) #UNK
#     return indexes

def indexesFromSentence(sentence):
    return [input_lang_word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang_index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

import rdkit
from rdkit import Chem

def can_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol)
        return smi
    except:
        print("output smiles is %s unvaild" % smi)
        return None

def evaluateRandomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0].replace(" ", ""))
        print('=', pair[1].replace(" ", ""))
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words).replace("<EOS>", "")
        output_sentence = can_smi(output_sentence)
        print('<', output_sentence)
        print('')

def evaluateAll(encoder, decoder, pairs):
    top1 = 0
    inputs = []
    predictions = []
    targets = []
    matches = []
    for pair in pairs:
        try:
            # print('>', pair[0].replace(" ", ""))
            # print('=', pair[1].replace(" ", ""))
            # print(pairs[0])
            # if "%" in pair[0].replace(" ", "") or "%" in pair[1].replace(" ", ""):
            #     continue
            # if 4 in pair[0].replace(" ", "") or 4 in pair[1].replace(" ", ""):
            #     continue
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            inputs.append(pair[0].replace(" ", ""))
            # print("output_words ", output_words)
            output_sentence = ''.join(output_words).replace("<EOS>", "")
            predictions.append(output_sentence)
            targets.append(pair[1].replace(" ", ""))

            output_sentence = can_smi(output_sentence)
            # print('<', output_sentence)
            # print('')
            if output_sentence == can_smi(pair[1].replace(" ", "")):
                print("正确", output_sentence)
                top1 += 1
                matches.append(1)
            else:
                matches.append(0)
        except Exception as E:
            print("APPEAR ERROR TOKEN", E)
            continue
    acc = top1/len(pairs) * 100
    print("the final acc is ", acc, " %")
    result_df = pd.DataFrame({"inputs": inputs, "targets": targets, "predictions": predictions, "matches":matches})
    result_df.to_csv("预测结果.csv")
    return acc

if __name__ == '__main__':
    encoder1 = torch.load('model/AGEs_cml.cel/encoder.pth').to(device)
    attn_decoder1 = torch.load('model/AGEs_cml.cel/decoder.pth').to(device)
    val_pairs = get_pairs('data/test/src-test.txt', 'data/test/tgt-test.txt')
    # val_pairs = get_pairs('data/AGEs_CML/src-val.txt', 'data/AGEs_CML/tgt-val.txt')
    # evaluateRandomly(encoder1, attn_decoder1, pairs, n=1)
    evaluateAll(encoder1, attn_decoder1, val_pairs)
