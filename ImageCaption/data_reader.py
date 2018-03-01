# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from PIL import Image
import paddle.v2 as paddle

__UNK__ = 'UNK'
__GO__ = 'GO'
__EOS__ = 'EOS'


# 保存词典
def save_dict(dict_path, word2id, id2word):
    f = open(dict_path + 'word2id.pkl', 'wb')
    pickle.dump(word2id, f)
    f.close()

    f = open(dict_path + 'id2word.pkl', 'wb')
    pickle.dump(id2word, f)
    f.close()


# 加载词典
def load_dict(dict_path):
    f = open(dict_path + 'word2id.pkl', 'rb')
    word2id = pickle.load(f)
    f.close()

    f = open(dict_path + 'id2word.pkl', 'rb')
    id2word = pickle.load(f)
    f.close()
    return word2id, id2word

# 存储转换的label
def save_converted_labels(filepath, converted_labels):
    f = open(filepath, 'wb')
    pickle.dump(converted_labels, f)
    f.close()

# 加载转换的label
def load_converted_labels(filepath):
    f = open(filepath, 'rb')
    labels = pickle.load(f)
    f.close()
    return labels


# preprocess the labels
def preprocess():

    token_file_path = 'data/Flickr8k_text/Flickr8k.token.txt'
    # 读取文件
    with open(token_file_path, 'r') as f:
        lines = f.readlines()
    lines = [x.strip('\n\r') for x in lines]

    # 分离出描述文字
    sentences = dict()
    for line in lines:
        elements = line.split('\t')
        img_name = elements[0].split('#')[0]
        if img_name not in sentences:
            sentences[img_name] = elements[1]

    # 构建词典
    sentence_words = dict()
    tokens = dict()
    for key in sentences.keys():
        words = sentences[key].lower().split(' ')
        if words[-1] == '.':
            words = words[:-1]
        sentence_words[key] = words
        for word in words:
            num = tokens.get(word, 0)
            tokens[word] = num + 1

    # vocabulary
    vocabulary = [__UNK__, __GO__, __EOS__] + sorted(tokens, key=tokens.get, reverse=True)

    # 词典/索引 转化
    words2id = {}
    id2words = {}
    for (i, v) in enumerate(vocabulary):
        words2id[v] = i
        id2words[i] = v

    # 将label转为数字label
    result = {}
    for key in sentence_words.keys():
        words = sentence_words[key]
        word_id = [words2id.get(x, 0) for x in words]
        result[key] = word_id

    if not os.path.exists('dict/'):
        os.mkdir('dict/')
    save_dict('dict/', words2id, id2words)
    save_converted_labels('data/converted_labels.pkl', result)


image_list_path = '/Users/Vic/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/ImageCaption/data/Flickr8k_text/'

# open image list
def get_image_list(is_train=True):

    if is_train:
        list_path = image_list_path + 'Flickr_8k.trainImages.txt'
    else:
        list_path = image_list_path + 'Flickr_8k.testImages.txt'
    with open(list_path, 'r') as f:
        lines = f.readlines()
    image_list = [x.rstrip('\n\r') for x in lines]

    return image_list


image_dir = '/Users/Vic/Downloads/Flicker8k_Dataset/'


# reader
def create_reader(is_train):

    image_list = get_image_list(is_train)
    num_samples = len(image_list)
    labels = load_converted_labels('data/converted_labels.pkl')

    def reader():
        for idx in xrange(num_samples):
            image_name = image_list[idx]
            label = labels[image_name]
            image_path = image_dir + image_name
            img = paddle.image.load_and_transform(
                filename=image_path,
                resize_size=224,
                crop_size=224,
                is_train=True,
                is_color=True
            ).flatten().astype('float32')
            yield img, [1] + label, label + [2]

    return reader

