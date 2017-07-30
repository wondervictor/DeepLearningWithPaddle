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
import gzip
import sys

STEP = 28
CLASS_DIM = 10


paddle.init(use_gpu=False, trainer_count=1)


def lstm(input, index):
    lstm_layer = paddle.networks.simple_lstm(
        name='lstm_%s' % index,
        size=64,
        input=input
    )
    return lstm_layer

x = []
for i in range(STEP):
    tmp_x = paddle.layer.data(
        name='x_%s' % i,
        type=paddle.data_type.dense_vector_sequence(STEP)
    )
    x.append(tmp_x)


def model(input_seqs):
    lstm_seqs = []
    for i in range(STEP):
        tmp_lstm = lstm(
            input=input_seqs[i],
            index=i
        )
        lstm_seqs.append(tmp_lstm)

    fc_1 = paddle.layer.fc(
        input=paddle.layer.concat(input=lstm_seqs),
        size=128,
        act=paddle.activation.Relu()
    )
    pool = paddle.layer.pooling(
        input=fc_1,
        pooling_type=paddle.pooling.Max()
    )

    output = paddle.layer.fc(
        input=pool,
        size=CLASS_DIM,
        act=paddle.activation.Softmax()
    )

    return output

output = model(x)


def train():

    label = paddle.layer.data(
        name='label',
        type=paddle.data_type.integer_value(CLASS_DIM)
    )

    loss = paddle.layer.classification_cost(
        input=output,
        label=label
    )

    parameters = paddle.parameters.create(loss)

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4)
    )

    trainer = paddle.trainer.SGD(
        cost=loss,
        parameters=parameters,
        update_equation=optimizer
    )

    feeding = dict()
    feeding['label'] = 28
    for i in range(STEP):
        feeding['x_%s' % i] = i

    def event_handler(event):

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                print ("\n pass %d, Batch: %d cost: %f metrics: %s" % (event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            test_reader = data_provider.create_reader('test', 10000)
            result = trainer.test(
                reader=paddle.batch(test_reader, batch_size=128),
                feeding=feeding)
            class_error_rate = result.metrics['classification_error_evaluator']
            print ("\nTest with Pass %d, cost: %s ratio: %f" % (event.pass_id, result.cost,class_error_rate))

    reader = data_provider.create_reader('train', 60000)
    trainer.train(
        paddle.batch(
            reader=reader,
            batch_size=128
        ),
        num_passes=200,
        event_handler=event_handler,
        feeding=feeding
    )


train()