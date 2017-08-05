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

import paddle.v2 as paddle
import sys
import gzip


NUM_CLASS = 10

relu = paddle.activation.Relu()

# CNN Layers
def cnn(image):
    conv_1 = paddle.networks.img_conv_group(
        input=image,
        num_channels=1,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[32, 32],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_2 = paddle.networks.img_conv_group(
        input=conv_1,
        num_channels=32,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_3 = paddle.networks.img_conv_group(
        input=conv_2,
        num_channels=64,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[128, 128],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_4 = paddle.networks.img_conv_group(
        input=conv_3,
        num_channels=128,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[256, 256],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )
    return conv_4


def bidirection_rnn(x):
    lstm_1 = paddle.networks.sim_lstm(
        input=x,
        size=128,
        act=relu
    )
    lstm_2 = paddle.networks.sim_lstm(
        input=x,
        size=128,
        act=relu,
        reversed=True
    )

    res = paddle.layer.fc(
        input=[lstm_1, lstm_2],
        size=NUM_CLASS+1,
        act=relu
    )

    return res

# RNN Layers
def rnn():


