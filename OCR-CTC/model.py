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
IMG_HEIGHT = 32
IMG_WIDTH = 100

relu = paddle.activation.Relu()

# CNN Layers
def cnn(image):

    conv_group_1 = paddle.networks.img_conv_group(
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
    # 16 x 50 x 32

    conv_group_2 = paddle.networks.img_conv_group(
        input=conv_group_1,
        num_channels=32,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )
    # 8 x 25 x 64

    conv_3 = paddle.layer.img_conv(
        input=conv_group_2,
        num_channels=64,
        num_filters=128,
        act=relu,
        stride=1
    )


    conv_4 = paddle.layer.img_conv(
        input=conv_3,
        num_channels=128,
        num_filters=128,
        act=relu,
        stride=1
    )

    pool_1 = paddle.layer.img_pool(
        input=conv_4,
        pool_size=1,
        pool_siez_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )
    # 4 x 25 x 64

    conv_5 = paddle.layer.img_conv(
        input=pool_1,
        num_channels=128,
        num_filters=256,
        act=relu,
        stride=1
    )

    pool_2 = paddle.layer.img_pool(
        input=conv_5,
        pool_size=1,
        pool_siez_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )

    conv_6 = paddle.layer.img_conv(
        input=pool_2,
        num_channels=256,
        num_filters=256,
        act=relu,
        stride=1
    )

    pool_3 = paddle.layer.img_pool(
        input=conv_6,
        pool_size=1,
        pool_siez_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )
    # 25 x 256
    return pool_3


def bidirection_rnn(x):
    lstm_fw = paddle.networks.sim_lstm(
        input=x,
        size=128,
        act=relu
    )
    lstm_bw = paddle.networks.sim_lstm(
        input=x,
        size=128,
        act=relu,
        reversed=True
    )

    res = paddle.layer.fc(
        input=[lstm_fw, lstm_bw],
        size=NUM_CLASS+1,
        act=paddle.activation.Linear()
    )

    return res


# RNN Layers
def rnn(x):
    x = bidirection_rnn(x)
    x = bidirection_rnn(x)
    return x


def feature_to_sequences(x):
    return paddle.layer.block_expand(
        input=x,
        num_channels=256,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=1
    )


def model(x):

    cnn_part = cnn(x)
    feature_sequence = feature_to_sequences(x)
    rnn_part = rnn(x)

    return rnn_part

def train():
    pass






