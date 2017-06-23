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
from paddle.trainer.PyDataProvider2 import *
import mnist_data as mnist


@provider(input_types={'image': dense_vector(784),
                       'label': integer_value(10)},
          cache=CacheType.CACHE_PASS_IN_MEM,
          should_shuffle=True)
def process(settings, filename):
    if filename == 'train':
        dataset = mnist.fetch_traingset()
    else:
        dataset = mnist.fetch_testingset()

    train_images = dataset['images']
    train_labels = dataset['labels']

    num_images = len(train_images)
    for i in range(num_images):
        yield {
            'image': train_images[i],
            'label': int(train_labels[i])
        }


@provider(input_types={'image': dense_vector(784)})
def predict_process(setting, filename):
    trainset = mnist.fetch_testingset()
    train_images = trainset['images']
    num_images = len(train_images)

    for i in range(num_images):
        yield {
            'image': train_images[i],
        }

