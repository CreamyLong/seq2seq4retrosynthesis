from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from prepare import tensorFromSentence, input_lang, output_lang
from config import MAX_LENGTH, SOS_token, EOS_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
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
            # print("top1", decoder_output.data.topk(1))
            # print("top2", decoder_output.data.topk(2))

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

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
        print("can smiles gets %s wrong" %  smi)
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
    for pair in pairs:
        print('>', pair[0].replace(" ", ""))
        print('=', pair[1].replace(" ", ""))
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words).replace("<EOS>", "")
        output_sentence = can_smi(output_sentence)
        print('<', output_sentence)
        print('')
        if output_sentence == pair[1].replace(" ", ""):
            top1 += 1
    acc = top1/len(pairs) * 100
    print(acc)
    return acc