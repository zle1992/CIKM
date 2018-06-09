#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
import jieba
import re

sys.path.append('utils/')
import config
# stopwords = [line.strip() for line in open(config.stopwords_path, 'r').readlines()]
# stopwords = [w.decode('utf8') for w in stopwords]
stopwords = []
# if config.cut_char_level:
#stopwords = [u'？', u'。', u'吗',u'，',u'的',u'怎么办',u'怎么']


def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # unit
    text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) +
                  ' kg ', text)        # e.g. 4kgs => 4 kg
    text = re.sub(r"(\d+)kg ", lambda m: m.group(1) +
                  ' kg ', text)         # e.g. 4kg => 4 kg
    text = re.sub(r"(\d+)k ", lambda m: m.group(1) +
                  '000 ', text)          # e.g. 4k => 4000
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c \+\+", "cplusplus", text)
    text = re.sub(r"c \+ \+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    text = re.sub(r"f#", "fsharp", text)
    text = re.sub(r"g#", "gsharp", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r",000", '000', text)
    text = re.sub(r"\'s", " ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r"pokemons", "pokemon", text)
    text = re.sub(r"pokémon", "pokemon", text)
    text = re.sub(r"pokemon go ", "pokemon-go ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r"insidefacebook", "inside facebook", text)
    text = re.sub(r"donald trump", "trump", text)
    text = re.sub(r"the big bang", "big-bang", text)
    text = re.sub(r"the european union", "eu", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" quaro ", " quora ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"the european union", " eu ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"₹", " rs ", text)      # 测试！
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text


def cut_single(x, cut_char_level):
    x = clean_text(x)
    res = []
    if cut_char_level:
        setence_seged = list(x.strip())
        # print(setence_seged)
    else:
        setence_seged = x.strip().split()

    for word in setence_seged:
        if word not in stopwords:
            res.append(word)

    return res


def moredata(data, random_state):
    q1 = data[data.label == 0][['q1', 'q1_cut']].sample(
        frac=0.2, random_state=random_state)
    q2 = data[data.label == 0][['q2', 'q2_cut']].sample(
        frac=0.2, random_state=random_state)

    data_new = pd.DataFrame()
    data_new['q1'] = q1.q1.values
    data_new['q1_cut'] = q1.q1_cut.values
    data_new['q2'] = q2.q2.values
    data_new['q2_cut'] = q2.q2_cut.values
    data_new['id'] = -1
    data_new['label'] = 0
    return data_new


def moredata2(data, random_state):

    q2 = data[data.label == 0][['q2', 'q2_cut']].sample(
        frac=0.2, random_state=random_state)

    q1 = data[data.label == 0][['q1', 'q1_cut']].sample(
        frac=0.2, random_state=random_state)

    data_new = pd.DataFrame()
    data_new['q1'] = q1.q1.values
    data_new['q1_cut'] = q1.q1_cut.values
    data_new['q2'] = q2.q2.values
    data_new['q2_cut'] = q2.q2_cut.values
    data_new['id'] = -1
    data_new['label'] = 0
    return data_new


def more(data, n):
    print('more_data-----')
    for i in range(n + 1):
        if i == 0:
            data1 = pd.DataFrame()
        else:
            data1 = data1.append(moredata(data, random_state=i))
            data1 = data1.append(moredata2(data, random_state=i))

    return data.append(data1)


def cut_word(data, feats):
    for f in feats:
        print('cut {0} done'.format(f))
        data[f + '_cut'] = data[f].map(lambda x: cut_single(x, config.cut_char_level))
    
    print(data.shape)
    return data

def read_cut_es():
    data1 = pd.read_table(config.origin_en_train, names=[
                          'q1_en', 'q1_es', 'q2_en', 'q2_es', 'label'])
    data1 = cut_word(data1, ['q1_es', 'q1_en', 'q2_es', 'q2_en'])

    data2 = pd.read_table(config.origin_es_train, names=[
                          'q1_es', 'q1_en', 'q2_es', 'q2_en', 'label'])
    data2 = cut_word(data2, ['q1_es', 'q1_en', 'q2_es', 'q2_en'])

    data = data1.append(data2)
    return data[['q1_es_cut','q2_es_cut','label']]

def read_cut_test():
    data3 = pd.read_table(config.origin_es_test, names=['q1_es', 'q2_es','label'])
    data3.label = -1
    data3 = cut_word(data3, ['q1_es', 'q2_es'])
    return data3[['q1_es_cut','q2_es_cut','label']]
def read_cut(path):
    if not os.path.exists(config.data_cut_hdf):
        data = cut_word(path, config.cut_char_level)
        data.to_hdf(config.data_cut_hdf, "data")
    data = pd.read_hdf(config.data_cut_hdf)
    return data
if __name__ == '__main__':
    path = config.origin_csv
    # read_data(path)
    cut_word(path)
