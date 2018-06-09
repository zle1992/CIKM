# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.path.append('utils/')
# sys.path.append('feature/')
import config

from CutWord import cut_single, cut_word


def read_words(words):
    words_all = []
    for word in words:
        words_all.extend(word)
    words_set = list(set(words_all))
    words_set = ['unknow'] + words_set
    words_set = words_set + ['pos', 'eos', 'padding']
    words_set = np.array(words_set)
    return words_set


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, 'rb') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def load_pre_train_embeddings(vocab, vectors,):
    vector_length = len(vocab)
    weights = np.zeros((vector_length, 300),   dtype='float32')
    cnt = 0
    # Normalization
    for index, word in enumerate(vocab):
        if word in vectors:
            weights[index] = vectors[word]
        else:
            weights[index] = np.random.random(size=weights.shape[1])
            cnt += 1
    print('vocab oov:{0}/{1}'.format(cnt, len(vocab)))
    return weights


def save_my_w2v_es():
    data1 = pd.read_table(config.origin_en_train, names=[
                          'q1_en', 'q1_es', 'q2_en', 'q2_es', 'label'])
    data1 = cut_word(data1, ['q1_es', 'q1_en', 'q2_es', 'q2_en'])
    print(data1.q1_es_cut)

    data2 = pd.read_table(config.origin_es_train, names=[
                          'q1_es', 'q1_en', 'q2_es', 'q2_en', 'label'])
    data2 = cut_word(data2, ['q1_es', 'q1_en', 'q2_es', 'q2_en'])

    data3 = pd.read_table(config.origin_es_test, names=['q1_es', 'q2_es'])
    data3 = cut_word(data3, ['q1_es', 'q2_es'])

    data4 = pd.read_table(config.origin_en_es, names=['q1_es', 'q1_en'])
    data4 = cut_word(data4, ['q1_es', 'q1_en'])

    words = list(data1['q1_es_cut']) + list(data1['q2_es_cut']) + \
        list(data2['q1_es_cut']) + list(data2['q2_es_cut']) + \
        list(data3['q1_es_cut']) + list(data3['q2_es_cut']) + list(data4['q1_es_cut'])

    vocab = read_words(words)
    # Read top n word vectors. Read all vectors when topn is 0
    vectors, iw, wi, dim = read_vectors(config.pre_w2v_es, 0)

    m = load_pre_train_embeddings(vocab, vectors)
    np.save(config.word_embed_es_vocab, vocab)
    np.save(config.word_embed_es_weight, m)

def load_pre_train_w2v_es():
    if not os.path.exists(config.word_embed_es_vocab):
        save_my_w2v_es()
    vocab = np.load(config.word_embed_es_vocab)
    vocab = {w: i for i, w in enumerate(vocab)}
    embed_weights = np.load(config.word_embed_es_weight)

    print('load embed_weights and vocab!')
    return vocab, embed_weights


if __name__ == '__main__':

    save_my_w2v_es()
