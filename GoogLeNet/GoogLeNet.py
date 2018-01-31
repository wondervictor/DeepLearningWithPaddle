# -*- coding:utf-8 -*-
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

import sys
import gzip
import paddle.v2 as paddle
import data_provider


paddle.init(use_gpu=False, trainer_count=2)

class_num = 100

image_size = 32 * 32 * 3


def inception(
        input,  # 输入
        name,   # inception 模块名
        channels,   # 输入通道
        filter_num1,    # 左侧1 1x1 卷积核
        filter_num2,    # 左侧2 1x1 卷积核
        filter_num3,    # 右侧2 1x1 卷积核
        filter_num4,    # 左侧2 3x3 卷积核
        filter_num5,    # 右侧2 5x5 卷积核
        filter_num6     # 右侧1 1x1 卷积核
):

    conv_1 = paddle.layer.img_conv(
        input=input,
        name=name+'conv_1',
        filter_size=1,
        num_channels=channels,
        num_filters=filter_num1,
        stride=1,
        padding=0
    )

    conv_2 = paddle.layer.img_conv(
        input=input,
        name=name+'conv_2',
        filter_size=1,
        num_filters=filter_num2,
        num_channels=channels,
        stride=1,
        padding=0
    )

    conv_3 = paddle.layer.img_conv(
        input=input,
        name=name+'conv_3',
        filter_size=1,
        num_filters=filter_num3,
        num_channels=channels,
        stride=1,
        padding=0
    )

    pool = paddle.layer.img_pool(
        input=input,
        name=name+'pool',
        pool_size=3,
        stride=1,
        padding=1,
        num_channels=channels,
        pool_type=paddle.MaxPool()
    )

    conv_3x3 = paddle.layer.img_conv(
        input=conv_2,
        name=name+'conv_3x3',
        filter_size=3,
        num_filters=filter_num4,
        padding=1,
        stride=1
    )

    conv_5x5 = paddle.layer.img_conv(
        input=conv_3,
        name=name+'conv_5x5',
        filter_size=5,
        num_filters=filter_num5,
        padding=2,
        stride=1
    )

    conv_1x1 = paddle.layer.img_conv(
        input=pool,
        name=name+'conv_1x1',
        filter_size=1,
        num_filters=filter_num6,
        padding=1,
        stide=1
    )

    conv_concat = paddle.layer.concat(
        name=name+'concat',
        input=[conv_1, conv_1x1, conv_3x3, conv_5x5]
    )

    return conv_concat



def inception_result(name, input, filters):
    avg_pool_1 = paddle.layer.img_pool(name=name+'avg_pool',
                                       input=inception_3,
                                       pool_type=paddle.AvgPool(),
                                       pool_size=5,
                                       stride=3)
    conv = paddle.layer.img_conv(name=name+'conv',
                                 input=avg_pool_1,
                                 filter_size=1,
                                 stride=1,
                                 num_filers=filters,
                                 padding=0)

    drop_outs = paddle.layer.dropout(input=conv, dropout_rate=0.4)

    fc = paddle.layer.fc(input=drop_outs, size=1000, act=paddle.activation.Linear())

    return fc


image = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(image_size))
label = paddle.layer.data(name='label', type=paddle.data_type.dense_vector(class_num))

# conv1

conv_1 = paddle.layer.img_conv(input=image,
                               name='conv_1',
                               filter_size=7,
                               num_filters=64,
                               num_channels=3,
                               stride=2,
                               padding=1)


pool_1 = paddle.layer.img_pool(input=conv_1,
                               name='pool_1',
                               pool_size=3,
                               stride=1,
                               pool_type=paddle.MaxPool())

conv_2 = paddle.layer.img_conv(input=image,
                               name='conv_1',
                               filter_size=7,
                               num_filters=64,
                               num_channels=3,
                               stride=2,
                               padding=1)

conv_3 = paddle.layer.img_conv(input=image,
                               name='conv_1',
                               filter_size=7,
                               num_filters=64,
                               num_channels=3,
                               stride=2,
                               padding=1)
pool_2 = paddle.layer.img_conv(input=image,
                               name='conv_1',
                               filter_size=7,
                               num_filters=64,
                               num_channels=3,
                               stride=2,
                               padding=1)


inception_1 = inception(input=pool_2,
                        name="incep_1",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_2 = inception(input=inception_1,
                        name="incep_2",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_3 = inception(input=inception_2,
                        name="incep_3",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_out_1 = inception_result(input=inception_3,
                                   name="out_1",
                                   filters=128)

cost_1 = paddle.layer.cross_entropy(input=inception_out_1, label=label)



inception_4 = inception(input=inception_3,
                        name="incep_4",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_5 = inception(input=inception_4,
                        name="incep_5",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_6 = inception(input=inception_5,
                        name="incep_6",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_out_2 = inception_result(input=inception_6,
                                   name="out_2",
                                   filters=128)

cost_2 = paddle.layer.cross_entropy(input=inception_out_2, label=label)

inception_7 = inception(input=inception_6,
                        name="incep_7",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_8 = inception(input=inception_7,
                        name="incep_8",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

inception_9 = inception(input=inception_8,
                        name="incep_9",
                        channels=0,
                        filter_num1=0,
                        filter_num2=0,
                        filter_num3=0,
                        filter_num4=0,
                        filter_num5=0,
                        filter_num6=0)

pool_3 = paddle.layer.img_pool(input=inception_9,
                               name='pool_3',
                               pool_type=paddle.AvgPool(),
                               pool_size=7,
                               stride=1)

inception_out_3 = inception_result(input=pool_3,
                                   name="out_3",
                                   filters=128)

cost_3 = paddle.layer.cross_entropy(input=inception_out_3, label=label)

parameters = paddle.parameters.create(cost_3)

momentum_optimizer = paddle.optimizer.Momentum(
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
    learning_rate=0.1 / 128.0,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
    learning_rate_schedule='discexp')

# Create trainer
trainer = paddle.trainer.SGD(cost=cost_3,
                             parameters=parameters,
                             update_equation=momentum_optimizer)


feeding={'image': 0,
         'label': 1}


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)
        reader_test = data_provider.data_reader('data/train', 10000)

        result = trainer.test(
            reader=paddle.batch(reader=reader_test, batch_size=16),
            feeding=feeding)
        print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

reader = data_provider.data_reader('data/train', 50000)
trainer.train(reader=paddle.batch(reader=reader, batch_size=16),
              num_passes=50,
              event_handler=event_handler,
              feeding=feeding)