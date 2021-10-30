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

from evaluate import evaluateRandomly, evaluateAll
from prepare import input_lang, output_lang, pairs
from train import trainIters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


from model import EncoderRNN, AttnDecoderRNN

# hidden_size = 256
encoder1 = torch.load('models/encoder.pth').to(device)

attn_decoder1 = torch.load('models/decoder.pth').to(device)

# evaluateRandomly(encoder1, attn_decoder1, pairs, n=1)
evaluateAll(encoder1, attn_decoder1, pairs)
