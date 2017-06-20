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

import paddle.trainer_config_helpers as paddle

"""
    LeNet
"""

is_predict=get_config_arg('is_predict', bool, False)

process = 'process'

test = 'data/test.list'
train = 'data/train.list'

if is_predict:
    process = 'predict_process'
    test = None
    train = 'predict.list'

paddle.define_py_data_sources2(
    train_list=train,
    test_list=test,
    module='data_provider',
    obj=process
)

batch_size = 16

if is_predict:
    batch_size = 1


paddle.settings(
    batch_size=batch_size,
    learning_method=paddle.MomentumOptimizer(),
    learning_rate=0.0001,
    regularization=paddle.L2Regularization(1e-4),
    gradient_clipping_threshold=20
)


image = paddle.data_layer(name='image', size=784)

conv_1 = paddle.img_conv_layer(input=image,
                               padding=1,
                               stride=1,
                               num_channels=1,
                               num_filters=6,
                               filter_size=5,
                               act=paddle.LinearActivation())

pool_1 = paddle.img_pool_layer(input=conv_1,
                               stride=2,
                               pool_size=2,
                               pool_type=paddle.MaxPooling())

conv_2 = paddle.img_conv_layer(input=pool_1,
                               stride=1,
                               num_filters=16,
                               filter_size=5,
                               act=paddle.LinearActivation()
                               )
pool_2 = paddle.img_pool_layer(input=conv_2,
                               stride=2,
                               pool_size=2,
                               pool_type=paddle.MaxPooling())

#conv_3 = paddle.img_conv_layer(input=pool_2)

fc_1 = paddle.fc_layer(input=pool_2, size=500, act=paddle.SigmoidActivation())
fc_2 = paddle.fc_layer(input=fc_1, size=10, act=paddle.SoftmaxActivation())

label = paddle.data_layer(name='label', size=10)

cost = paddle.classification_cost(input=fc_2, label=label)

paddle.outputs(cost)


