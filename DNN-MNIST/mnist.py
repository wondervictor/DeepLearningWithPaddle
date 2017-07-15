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

import dnn
import os

import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np


def train():
    dnn.train(50)


def test(images):
    return dnn.predict(images, "output/model.tar.gz")


def get_num(arr):
    num = 0
    max_value = arr[0]
    for i in range(1, 10):
        if max_value < arr[i]:
            num = i
            max_value = arr[i]

    return num, max_value


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def read_image(path):
    num = mpimg.imread(path)
    num = rgb2gray(num)
    num = num.reshape([-1])
    image = num.tolist()
    result = test([image])[0]
    print(result)
    num, confidence = get_num(result)

    print("Num=%d Confidence: %.10f\n" % (num, confidence))

read_image('/Users/vic/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/DNN-MNIST/4.png')


