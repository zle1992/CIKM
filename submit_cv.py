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
from Feats import data_2id, add_hum_feats
from CutWord import cut_word
from help import get_X_Y_from_df
from CutWord import read_cut_test
import pandas as pd
import time 

def load_data():
   
    print('load data')
    data = read_cut_test()  #cut word
    data = data_2id(data,['q1_es_cut','q2_es_cut'])  # 2id

    data = add_hum_feats(data,config.test_feats) #生成特征并加入

    return data


def make_test_cv_data(X_dev, model_name, epoch_nums, kfolds):
    mean_epoch = False
    test_df = pd.DataFrame()
    S_test = np.zeros((X_dev[0].shape[0], epoch_nums))
    for epoch_num in range(epoch_nums):
        for kf in range(1, kfolds + 1):
            print('kf: ', kf)
            print('epoch_num: ', epoch_num + 1)
            model = load_model(config.stack_path+"_%s_%s.h5" %
                               (model_name, epoch_num), custom_objects={"softmax": softmax})
            pred = model.predict(X_dev, batch_size=config.batch_size)

            S_test[:, epoch_num] += pred[:, 1]
        S_test[:, epoch_num] /= kfolds

        test_df['epoch_%s' % (epoch_num)] = S_test[:, epoch_num]
        test_df.to_csv(config.stack_path+'test_%s.csv' % (model_name),
                       index=False,)
        if mean_epoch:
            pred = np.mean(S_test, axis=1)
        else:
            pred = S_test[:,epoch_num]
        return pred


def do_cv_test():

    model_name = 'cnn'
    epoch_nums = 5
    kfolds = 5
    out_path = 'submit/{0}_{1}.txt'.format(model_name,time.time())
    data = load_data()
    X, _ = get_X_Y_from_df(data, False)
    if config.feats == []:
        X = X[:2]
    pred = make_test_cv_data(X, model_name, epoch_nums, kfolds)
    data['label'] =  pred
    data['label'].to_csv(out_path, index=False, header=None,)


if __name__ == '__main__':
    #main(sys.argv[1], sys.argv[2])
    
    
    do_cv_test()
    # main_test(sys.argv[1])
