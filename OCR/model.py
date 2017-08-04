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


# cnn layers
def model(x):

    relu = paddle.activation.Relu()
    conv_1 = paddle.networks.img_conv_group(
        input=x,
        num_channels=3,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_2 = paddle.networks.img_conv_group(
        input=conv_1,
        num_channels=3,
        pool_size=64,
        pool_stride=2,
        conv_num_filter=[128, 128],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_3 = paddle.networks.img_conv_group(
        input=conv_2,
        num_channels=3,
        pool_size=128,
        pool_stride=2,
        conv_num_filter=[256, 256],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_4 = paddle.networks.img_conv_group(
        input=conv_1,
        num_channels=3,
        pool_size=256,
        pool_stride=2,
        conv_num_filter=[512, 512],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    flatten = paddle.layer.fc(
        input=conv_4,
        size=512,
        act=relu
    )

    fc_1 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_2 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_3 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_4 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    output = paddle.layer.concat(
        input=[fc_1, fc_2, fc_3, fc_4]
    )

    return output


IMAGE_SIZE = 32*80*3
LABEL_SIZE = 10


def train():
    paddle.init(use_gpu=False, trainer_count=2)

    x = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(IMAGE_SIZE)
    )

    label = paddle.layer.data(
        name='label',
        type=paddle.data_type.integer_value_sequence(LABEL_SIZE)
    )

    output = model(x)

    loss = paddle.layer.classification_cost(input=output, label=label)

    parameters = paddle.parameters.create(loss)















