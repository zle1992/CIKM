#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import numpy as np
from keras.activations import softmax
from keras import backend
# Model Load
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id,add_hum_feats
from CutWord import read_cut_test
from help import get_X_Y_from_df



def main(model_path):
    out_path = 'submit/{0}.txt'.format(model_path.split('/')[-1])
    print('load data')
    data = read_cut_test()  #cut word
    data = data_2id(data,['q1_es_cut','q2_es_cut'])  # 2id

    data = add_hum_feats(data,config.test_feats) #生成特征并加入
    X, _ = get_X_Y_from_df(data, False)
    if config.feats==[]:
        X = X[:2]
    print('load model and predict')
    model = load_model(model_path, custom_objects={"softmax": softmax})
    test_pred = model.predict(X, batch_size=config.batch_size)
    print(test_pred)
    data['label'] = test_pred[:, 1]
    data['label'].to_csv(out_path, index=False, header=None,)

if __name__ == '__main__':
    main(sys.argv[1])
