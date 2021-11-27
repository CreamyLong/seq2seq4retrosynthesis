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
from config import MAX_LENGTH, SOS_token, EOS_token, device

import pickle

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# def readLangs(lang1, lang2, reverse=False):
#     print("Reading lines...")
#
#     # Read the file and split into lines
#     lines = open('./%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
#
#     # Split every line into pairs and normalize
#     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
#
#     # Reverse pairs, make Lang instances
#     if reverse:
#         pairs = [list(reversed(p)) for p in pairs]
#         input_lang = Lang(lang2)
#         output_lang = Lang(lang1)
#     else:
#         input_lang = Lang(lang1)
#         output_lang = Lang(lang2)
#
#     return input_lang, output_lang, pairs



def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    src_lines = open('data/AGEs_cml.cel/src-train.txt', encoding='utf-8').read().strip().split('\n')
    tgt_lines = open('data/AGEs_cml.cel/tgt-train.txt', encoding='utf-8').read().strip().split('\n')
    #
    # src_lines = open('data/AGEs_2.3.4/src-val.txt', encoding='utf-8').read().strip().split('\n')
    # tgt_lines = open('data/AGEs_2.3.4/tgt-val.txt', encoding='utf-8').read().strip().split('\n')
    #
    # src_lines = open('data/test/src-test.txt', encoding='utf-8').read().strip().split('\n')
    # tgt_lines = open('data/test/tgt-test.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = []
    for i in range(len(src_lines)):
      lines = [src_lines[i], tgt_lines[i]]
      pairs.append(lines)


    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# def indexesFromSentence(lang, sentence):
#     indexes = []
#     for word in sentence.split(' '):
#         if word in lang.word2index:
#             index = lang.word2index[word]
#             indexes.append(index)
#         else:
#             indexes.append(2) #UNK
#     return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang,output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    print("Read %s sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    f_save = open('input_lang_word2index.pkl', 'wb')
    pickle.dump(input_lang.word2index, f_save)
    f_save.close()

    f_save = open('input_lang_index2word.pkl', 'wb')
    pickle.dump(input_lang.index2word, f_save)
    f_save.close()

    f_save = open('output_lang_word2index.pkl', 'wb')
    pickle.dump(output_lang.word2index, f_save)
    f_save.close()

    f_save = open('output_lang_index2word.pkl', 'wb')
    pickle.dump(output_lang.index2word, f_save)
    f_save.close()

    return input_lang, output_lang, pairs


# input_lang, output_lang, pairs = prepareData('src', 'tgt', False)

if __name__ == '__main__':

    input_lang, output_lang, pairs = prepareData('src', 'tgt', False)
    print(random.choice(pairs))
    f_read = open('input_lang_word2index.pkl', 'rb')
    dict2 = pickle.load(f_read)
    print(dict2)
    f_read.close()

