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
import struct


def read_image_files(filename, num):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    # 前四个32位integer为以下参数
    # >IIII 表示使用大端法读取
    magic, numImage, numRows, numCols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    image_sets = []
    for i in range(num):
        images = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        images = np.array(images)
        images = images/255.0
        images = images.tolist()
        # if i == 6:
        #     print ','.join(['%s'%x for x in images])
        image_sets.append(images)
    bin_file.close()
    return image_sets


def read_label_files(filename):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    magic, nums = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = struct.unpack_from('>%sB'%nums, buf, index)
    bin_file.close()
    labels = np.array(labels)
    return labels


def fetch_traingset():
    image_file = 'data/train-images-idx3-ubyte'
    label_file = 'data/train-labels-idx1-ubyte'
    images = read_image_files(image_file,60000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}


def fetch_testingset():
    image_file = 'data/t10k-images-idx3-ubyte'
    label_file = 'data/t10k-labels-idx1-ubyte'
    images = read_image_files(image_file,10000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}


def create_reader(filename, n):
    def reader():
        if filename == 'train':
            dataset = fetch_traingset()
        else:
            dataset = fetch_testingset()
        for i in range(n):
            yield dataset['images'][i], dataset['labels'][i]

    return reader
