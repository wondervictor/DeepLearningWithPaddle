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
import gzip
import sys
import data_provider
import numpy as np
import matplotlib.pyplot as plt


def param():
    return paddle.attr.Param(
        initial_std=0.01,
        initial_mean=0
    )


def encoder(x_):
    x_ = paddle.layer.fc(
        input=x_,
        size=512,
        act=paddle.activation.Sigmoid(),
        param_attr=param(),
        bias_attr=param()
    )
    x_ = paddle.layer.fc(
        input=x_,
        size=256,
        act=paddle.activation.Relu(),
        param_attr=param(),
        bias_attr=param()
    )
    x_ = paddle.layer.fc(
        input=x_,
        size=128,
        act=paddle.activation.Relu(),
        param_attr=param(),
        bias_attr=param()
    )
    return x_


def decoder(x_):
    x_ = paddle.layer.fc(
        input=x_,
        size=128,
        act=paddle.activation.Sigmoid(),
        param_attr=param(),
        bias_attr=param()
    )
    x_ = paddle.layer.fc(
        input=x_,
        size=256,
        act=paddle.activation.Relu(),
        param_attr=param(),
        bias_attr=param()
    )
    x_ = paddle.layer.fc(
        input=x_,
        size=512,
        act=paddle.activation.Relu(),
        param_attr=param(),
        bias_attr=param()
    )
    return x_


def output(x_):
    return paddle.layer.fc(
        input=x_,
        size=784,
        act=paddle.activation.Relu(),
        param_attr=param(),
        bias_attr=param()
    )

paddle.init(use_gpu=False, trainer_count=1)
x = paddle.layer.data(
    name='x',
    type=paddle.data_type.dense_vector(784)
)

y = encoder(x)
y = decoder(y)
y = output(y)


def train():

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4)
    )
    loss = paddle.layer.mse_cost(label=x, input=y)

    parameters = paddle.parameters.create(loss)

    trainer = paddle.trainer.SGD(
        cost=loss,
        parameters=parameters,
        update_equation=optimizer
    )

    feeding = {'x': 0}

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

    reader = data_provider.create_reader('train', 60000)
    trainer.train(
        paddle.batch(
            reader=reader,
            batch_size=128
        ),
        feeding=feeding,
        num_passes=20,
        event_handler=event_handler
    )


def show(origin, pred):
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(origin[i], (28, 28)))
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()


def test(model_path):
    with gzip.open(model_path, 'r') as openFile:
        parameters = paddle.parameters.Parameters.from_tar(openFile)
    testset = data_provider.fetch_testingset()['images'][:10]
    # 使用infer进行预测
    result = paddle.infer(
        input=testset,
        parameters=parameters,
        output_layer=y,
        feeding={'x': 0}
    )
    


if __name__ == '__main__':
    #test('output/params_pass_18.tar.gz')
    train()

