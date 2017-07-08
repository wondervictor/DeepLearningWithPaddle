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


import paddle.v2 as paddle
import data_provider
import sys
import gzip

paddle.init(use_gpu=False, trainer_count=2)

image_size = 32 * 32 * 3


image = paddle.layer.data(name='image',  type=paddle.data_type.dense_vector(image_size))

conv_1 = paddle.layer.img_conv(name='conv_1',
                               input=image,
                               filter_size=11,
                               num_filters=96,
                               num_channels=3,
                               act=paddle.activation.Relu(),
                               padding=5,
                               stride=1)

lrn_conv_1 = paddle.layer.img_cmrnorm(name='norm_1',
                                      input=conv_1,
                                      size=5,
                                      scale=0.0001,
                                      power=0.75)

pool_1 = paddle.layer.img_pool(name='pool_1',
                               input=lrn_conv_1,
                               pool_type=paddle.pooling.Max(),
                               stride=2,
                               pool_size=2)

conv_2 = paddle.layer.img_conv(name='conv_2',
                               input=pool_1,
                               filter_size=5,
                               num_filters=256,
                               num_channels=96,
                               act=paddle.activation.Relu(),
                               stride=1,
                               padding=2)

lrn_conv_2 = paddle.layer.img_cmrnorm(name='norm_2',
                                      input=conv_2,
                                      size=5,
                                      scale=0.0001,
                                      power=0.75)

pool_2 = paddle.layer.img_pool(name='pool_2',
                               input=conv_2,
                               pool_type=paddle.pooling.Max(),
                               stride=2,
                               pool_size=2)

conv_3 = paddle.layer.img_conv(name='conv_3',
                               input=pool_2,
                               filter_size=3,
                               num_filters=384,
                               num_channels=256,
                               act=paddle.activation.Relu(),
                               stride=1,
                               padding=1)

conv_4 = paddle.layer.img_conv(name='conv_4',
                               input=conv_3,
                               filter_size=3,
                               num_filters=384,
                               num_channels=384,
                               act=paddle.activation.Relu(),
                               stride=1,
                               padding=1)

conv_5 = paddle.layer.img_conv(name='conv_5',
                               input=conv_4,
                               filter_size=3,
                               num_filters=256,
                               num_channels=384,
                               act=paddle.activation.Relu(),
                               stride=1,
                               padding=1)

pool_3 = paddle.layer.img_pool(name='pool_3',
                               input=conv_5,
                               pool_type=paddle.pooling.Max(),
                               stride=2,
                               pool_size=2)

fc_1 = paddle.layer.fc(name='fc_1',
                       input=pool_3,
                       size=4096,
                       act=paddle.activation.Relu(),
                       layer_attr=paddle.attr.Extra(drop_rate=0.5))

fc_2 = paddle.layer.fc(name='fc_2',
                       input=fc_1,
                       size=4096,
                       act=paddle.activation.Relu(),
                       layer_attr=paddle.attr.Extra(drop_rate=0.5))

fc_3 = paddle.layer.fc(name='fc_3',
                       input=fc_2,
                       size=10,
                       act=paddle.activation.Softmax())

label = paddle.layer.data(name='label',type=paddle.data_type.integer_value(10))

cost_layer = paddle.layer.classification_cost(input=fc_3, label=label)

parameters = paddle.parameters.create(cost_layer)


optimizer = paddle.optimizer.Momentum(momentum=0.9,
                                      regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
                                      learning_rate=0.1 / 128.0,
                                      learning_rate_decay_a=0.1,
                                      learning_rate_decay_b=50000 * 100,
                                      learning_rate_schedule='discexp')

trainer = paddle.trainer.SGD(parameters=parameters,update_equation=optimizer,cost=cost_layer)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "\nPass: %d Batch: %d [Cost: %f ][%s]\n" % (event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):

        #save parameters

        with gzip.open('output/params.tar.gz', 'w') as f:
            parameters.to_tar(f)

        # test
        feeding = {'image': 0,
                   'label': 1}
        filepath = ""
        result = trainer.test(reader=paddle.batch(reader=data_provider.data_reader(filepath, 0), batch_size=128),
                              feeding=feeding)
        print "\nTest Result: [Cost: %f] [%s] " % (result.cost, result.metrics)

feeding = {'image': 0,
           'label': 1}
file_path = '/Users/vic/Dev/DeepLearning/Paddle/VGG-CIFAR/Images/cifar-10-batches-py/data_batch_1'
reader = data_provider.data_reader(file_path, 0)
trainer.train(num_passes=10, reader=paddle.batch(reader, batch_size=128), event_handler=event_handler, feeding=feeding)
