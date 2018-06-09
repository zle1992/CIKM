#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os 
import pandas as pd
import numpy as np
from collections import defaultdict


sys.path.append('utils/')
import config
from w2v import load_pre_train_w2v_es
from CutWord import cut_word,more
from feat_gen import *


vocab, embed_weights = load_pre_train_w2v_es()



def padding_id(ids, padding_token=0, padding_length=None):
    if len(ids) > padding_length:
        return ids[:padding_length]
    else:
        return ids + [padding_token] * (padding_length - len(ids))


def word2id(contents, word_voc):
    ''' contents  list
    '''
#     contents = str(contents)
#     contents = contents.split()

    ids = [word_voc[c] if c in word_voc else len(word_voc) for c in contents]

    return padding_id(ids, padding_token=0, padding_length=config.word_maxlen)





def data_2id(data,feats):
    for f in feats:
        data[f+'_id'] = data[f].map(lambda x: word2id(x, vocab))
    return data


    

def human_feats(data,outpath,):
   
    
    data['q1_es_cut'] = data['q1_es_cut'].map(lambda x: ' '.join(x))
    data['q2_es_cut'] = data['q2_es_cut'].map(lambda x: ' '.join(x))
    
    print(data.columns)
    data = magic1(data)
    print(data.columns)
    return data

def load_hum_feats(data,path):

    if os.path.exists(path):
        return pd.read_hdf(path)
    else:
        data=human_feats(data,path)
        data.to_hdf(path,'data')
        return data

def add_hum_feats(data,path):
    if config.feats==[]:
            data['magic_feat'] = 0
    else:
        df1 = load_hum_feats(data,path)
        data['magic_feat'] = list(df1.values)
    return data

# def load_final_df(data,hdf_path):
#     if not os.path.exists(hdf_path):
#         data = data_2id(data)
#         if config.feats==[]:
#             data['magic_feat'] = 0
#         else:
#             df1 = human_feats(data)
#             data['magic_feat'] = list(df1.values)
#         data.to_hdf(hdf_path, "data")
#     else:
#         data = pd.read_hdf(hdf_path)
#     return data
# # def load_final_train_df(path):

#     data = data_2id(data)
#     df1 = human_feats(config.origin_csv)
#     data['magic_feat'] = list(df1.values)
#     # data.to_hdf(config.data_feat_hdf_train, "data")
#     # else:
#     #     data = pd.read_hdf(config.data_feat_hdf_)
#     return data

# def load_final_train_df(path):
    
#     data = cut_word(config.origin_csv)
#     data = data_2id(data)
#     data = f1(data)
#     # data.to_hdf(config.data_feat_hdf, "data")
#     # else :
#     #     data = pd.read_hdf(config.data_feat_hdf)
#     return data
def load_final_test_df(path):

    data = cut_word(path,config.cut_char_level)
    data = data_2id(data)
    # df1 = human_feats(path)
    # data['magic_feat'] = list(df1.values)
    return data

if __name__ == '__main__':

    load_final_data(config.data_cut_hdf)