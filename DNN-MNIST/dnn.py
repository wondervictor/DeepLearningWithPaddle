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
import data_provider
import sys
import gzip

paddle.init(use_gpu=False, trainer_count=1)

image_dim = 28*28
class_dim = 10

image = paddle.layer.data(
    name='image',
    type=paddle.data_type.dense_vector(image_dim)
)

def network(input):
    # fully connected hidden layers
    fc_1 = paddle.layer.fc(
        input=input,
        size=784,
        act=paddle.activation.Sigmoid()
    )

    fc_2 = paddle.layer.fc(
        input=fc_1,
        size=256,
        act=paddle.activation.Sigmoid()
    )

    fc_3 = paddle.layer.fc(
        input=fc_2,
        size=64,
        act=paddle.activation.Sigmoid()
    )

    # output layer
    output_layer = paddle.layer.fc(
        input=fc_3,
        size=class_dim,
        act=paddle.activation.Softmax()
    )
    return output_layer


def train(passes):

    output_layer = network(image)
    # cost
    label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(class_dim))
    cost = paddle.layer.classification_cost(input=output_layer, label=label)

    parameters = paddle.parameters.create(cost)

    print(parameters.keys())

    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp'
    )

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=momentum_optimizer
    )

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            test_reader = data_provider.create_reader('test', 10000)

            result = trainer.test(paddle.batch(reader=test_reader, batch_size=128), feeding=feeding)
            class_error_rate = result.metrics['classification_error_evaluator']
            with open('output/error', 'a+') as f:
                f.write('%f\n' % class_error_rate)

            print("\nTest with Pass %d, %f" % (event.pass_id, class_error_rate))

    reader = data_provider.create_reader('train', 60000)

    feeding = {'image': 0,
               'label': 1}

    trainer.train(
        reader=paddle.batch(reader=reader, batch_size=128),
        num_passes=passes,
        event_handler=event_handler,
        feeding=feeding
    )


def predict(x, model_path):

    output_layer = network(image)
    with gzip.open(model_path, 'r') as openFile:
        parameters = paddle.parameters.Parameters.from_tar(openFile)

    result = paddle.infer(input=x, parameters=parameters, output_layer=output_layer, feeding={'image': 0})

    return result


