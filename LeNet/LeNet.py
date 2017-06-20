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

is_predict=paddle.get_config_arg('is_predict', bool, False)

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


