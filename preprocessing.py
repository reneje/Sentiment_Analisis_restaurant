#preprocessing
import pandas as pd
import numpy as np
import string

from string import punctuation
from sklearn.utils import shuffle

kamus = open("kamus_alay.txt").read().split("\n")
dict_alay = {}
for i in kamus:
    split_kamus = i.split(',')
    dict_alay[split_kamus[0]] = split_kamus[1]
    
def load_dataset():
    df_train = pd.read_csv("train_data_restaurant.tsv", delimiter='\t', header=None)
    df_train= shuffle(df_train)
    df_test = pd.read_csv("test_data_restaurant.tsv", delimiter='\t', header=None)
    df_test=shuffle(df_test)
    return df_train,df_test

# def build_vocab(sentences, verbose =  True):
#     """
#     :param sentences: list of list of words
#     :return: dictionary of words and their count
#     """
# #     sentences=str(sentences)
#     vocab = {}
#     for sentence in sentences:
#         for word in sentence:
#             try:
#                 vocab[word] += 1
#             except KeyError:
#                 vocab[word] = 1
#     return vocab

def clean_word(text):
    # white_list = string.ascii_letters + string.digits+' '
    # list_train = list(text)
    # char = build_vocab(list_train)
    symbols = punctuation#''.join([c for c in char if not c in white_list])
    isolate_dict = {ord(c):f' {c} ' for c in symbols}
    text = text.translate(isolate_dict)
    return text

def clean_punct(x):
    return ''.join([c for c in x if c not in punctuation])

def normalization(text):
    x_split = text.lower()
    x_split = x_split.split()
    k = []
    for i in x_split:
        if i in dict_alay.keys():
            k.append(dict_alay[i])
        else:
            k.append(i)
    return " ".join(k)
