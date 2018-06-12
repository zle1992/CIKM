#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 256
word_maxlen = 40




 
                
model_dir = '../model_dir'
stopwords_path = 'data/jieba/stops.txt'
origin_en_train = 'data/origin_data/cikm_english_train_20180516.txt'
origin_es_train ='data/origin_data/cikm_spanish_train_20180516.txt'
origin_es_test = 'data/origin_data/cikm_test_a_20180516.txt'
origin_en_es=   'data/origin_data/cikm_unlabel_spanish_train_20180516.txt'

pre_w2v_es = 'data/fast_text_vectors_wiki.es.vec/wiki.es.vec'
pre_w2v_es = 'data/fast_text_vectors_wiki.en.vec/wiki.en.vec'


word_embed_es_vocab = 'data/word_embed_es_vocab.npy'
word_embed_es_weight = 'data/word_embed_es_weight.npy'
data_augment = True

stack_path = 'data/stack/'

feats =[]





use_pre_train = True
cut_char_level = False


data_cut_hdf ='data/cache/train_cut_word.hdf'
train_feats = 'data/cache/train_feats_word.hdf'
test_feats = 'data/cache/test_feats_word.hdf'
data_feat_hdf = 'data/cache/train_magic_word.hdf'

train_df= 'data/cache/train_magic_word_train_f{0}.hdf'.format(len(feats))
dev_df = 'data/cache/train_magic_word_more_dev_f{0}.hdf'.format(len(feats))

word_embed_weight = word_embed_es_weight
# w2v_content_word_model = 'data/my_w2v/train_word.model'




















