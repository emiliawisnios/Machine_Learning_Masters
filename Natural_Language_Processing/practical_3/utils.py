"""Implementation based on https://github.com/pcyin/pytorch_nmt and 
Stanford CS224 2019 class.
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
     
    Args:
        sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
        pad_token (str): padding token
    
    Returns:
        sents_padded (list[list[str]]): 
            list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, 
            such that each sentences in the batch now has equal length.
    """
    sents_padded = []

    # YOUR CODE HERE
    longest_sentence_len = 0
    for sentence in sents:
        longest_sentence_len = max(len(sentence), longest_sentence_len)

    for sentence in sents:
        sentence_ = sentence + ([pad_token] * (longest_sentence_len - len(sentence)))
        sents_padded.append(sentence_)

    # END YOUR CODE

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
     file_path (str): path to file containing corpus
     source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by 
    length (largest to smallest).
    
    Arguments:
        data (list of (src_sent, tgt_sent)): list of tuples containing 
            source and target sentence
        batch_size (int): batch size
        shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents