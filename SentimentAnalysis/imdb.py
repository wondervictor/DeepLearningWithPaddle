# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2017 Vic Chan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import cPickle as pickle
import random

# dict_dim = 102100

class Imdb(object):

    def __init__(self, dataset_path):
        f = open(dataset_path, 'rb')
        trainset = np.array(pickle.load(f))
        testset = np.array(pickle.load(f))
        f.close()

        trainset = zip(trainset[0], trainset[1])
        testset = zip(testset[0], testset[1])
        # shuffle 数据集顺序
        random.shuffle(trainset)
        random.shuffle(testset)
        self._trainset = trainset
        self._testset = testset

    def create_reader(self, type):

        def reader():
            if type == 'train':
                dataset = self._trainset
            else:
                dataset = self._testset
            for i in range(25000):
                yield dataset[i][0], dataset[i][1]

        return reader


def test():
    dataset = Imdb('data/imdb.pkl')
    reader = dataset.create_reader('test')
    reader()


#test()


