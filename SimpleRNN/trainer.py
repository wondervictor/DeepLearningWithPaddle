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
import data_provider


def network(x):
    recurrent = paddle.networks.simple_lstm(input=x, size=10, act=paddle.activation.Relu())

    fc_1 = paddle.layer.fc(input=recurrent, size=1, act=paddle.activation.Linear())

    output = paddle.layer.last_seq(input=fc_1)

    return output


def train(x_, model_path, is_predict=False):

    paddle.init(use_gpu=False, trainer_count=1)

    TIME_STEP = 10

    x = paddle.layer.data(
        name='x',
        type=paddle.data_type.dense_vector_sequence(TIME_STEP)
    )

    output = network(x)


    if not is_predict:

        label = paddle.layer.data(
            name='y',
            type=paddle.data_type.dense_vector(
                dim=1
            )
        )

        loss = paddle.layer.mse_cost(input=output, label=label)

        parameters = paddle.parameters.create(loss)

        optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4)
        )

        trainer = paddle.trainer.SGD(cost=loss,
                                     parameters=parameters,
                                     update_equation=optimizer)
        feeding = {'x': 0, 'y': 1}

        def event_handler(event):

            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 50 == 0:
                    print ("\n pass %d, Batch: %d cost: %f" % (event.pass_id, event.batch_id, event.cost))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            if isinstance(event, paddle.event.EndPass):
                # save parameters
                feeding = {'x': 0,
                           'y': 1}
                with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                    parameters.to_tar(f)
                filepath = 'data/test.data'
                result = trainer.test(
                    reader=paddle.batch(data_provider.data_reader(filepath), batch_size=16),
                    feeding=feeding)
                print ("\nTest with Pass %d, cost: %s" % (event.pass_id, result.cost))

        train_file_path = 'data/train.data'

        reader = data_provider.data_reader(train_file_path)

        trainer.train(
            paddle.batch(reader=reader, batch_size=128),
            num_passes=200,
            event_handler=event_handler,
            feeding=feeding
        )
    else:
        with gzip.open(model_path, 'r') as openFile:
            parameters = paddle.parameters.Parameters.from_tar(openFile)
        result = paddle.infer(input=x_, parameters=parameters, output_layer=output, feeding={'x':0})

        return result




