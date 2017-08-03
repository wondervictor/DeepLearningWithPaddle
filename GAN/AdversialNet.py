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

"""
    Generative Adversial Nets
    
"""

import paddle.v2 as paddle
import gzip
import sys
import numpy as np
# from paddle.trainer.config_parser import parse_config
# from paddle.trainer.config_parser import logger
# import py_paddle.swig_paddle as api


relu = paddle.activation.Relu()
sigmoid = paddle.activation.Sigmoid()


def generator(z):

    fc_1 = paddle.layer.fc(
        input=z,
        size=128,
        act=relu
    )

    fc_2 = paddle.layer.fc(
        input=fc_1,
        size=256,
        act=relu
    )
    # output the image
    fc_3 = paddle.layer.fc(
        input=fc_2,
        size=784,
        act=sigmoid
    )

    return fc_3


def discriminator(x):

    fc_1 = paddle.layer.fc(
        input=x,
        size=256,
        act=relu
    )

    fc_2 = paddle.layer.fc(
        input=fc_1,
        size=128,
        act=relu
    )

    # output the probability
    fc_3 = paddle.layer.fc(
        input=fc_2,
        size=1,
        act=sigmoid
    )
    return fc_3


image = paddle.layer.data(
    name='image',
    type=paddle.data_type.dense_vector(784)
)

noise = paddle.layer.data(
    name='noise',
    type=paddle.data_type.dense_vector(100)
)


def train():
    fake_sample = generator(noise)
    d_real = discriminator(image)
    d_fake = discriminator(fake_sample)

    d_loss_1 = paddle.layer.addto(input=d_real, act=paddle.activation.Log(), bias=False)

    parameters = paddle.parameters.create(d_loss_1)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                print ("\n pass %d, Batch: %d cost: %f"
                       % (event.pass_id, event.batch_id, event.cost))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)










def generate():
    pass

