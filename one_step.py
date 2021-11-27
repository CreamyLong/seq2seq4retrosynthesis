
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
from evaluate import evaluate, can_smi
from prepare import input_lang, output_lang, pairs
from model import EncoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_step(encoder, decoder, input):

    output_words, attentions = evaluate(encoder, decoder, input)
    output_sentence = ''.join(output_words).replace("<EOS>", "")
    output_sentence = can_smi(output_sentence)

    return output_sentence

encoder1 = torch.load('model/AGEs_2.3.4/encoder.pth').to(device)
attn_decoder1 = torch.load('model/AGEs_2.3.4/decoder.pth').to(device)

# for pair in pairs:
#     print(pair[0])
#     output = one_step(encoder1, attn_decoder1, pair[0])
#     print(output)


input ="C N C C ( O ) C 1 = C C ( O ) = C ( O ) C = C 1 . O C C 1 O C ( O ) C ( O ) C ( O ) C 1 O"
output = one_step(encoder1, attn_decoder1, input)
print("结果为", output)

