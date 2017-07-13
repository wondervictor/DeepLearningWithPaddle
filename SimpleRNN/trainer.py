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

paddle.init(use_gpu=False, trainer_count=1)

TIME_STEP = 10

x = paddle.layer.data(
    name='x',
    type=paddle.data_type.dense_vector(
        dim=TIME_STEP,
        seq_type=1
    )
)

label = paddle.layer.data(
    name='y',
    type=paddle.data_type.dense_vector(
        dim=1
    )
)

recurrent = paddle.v2.recurrent(
    input=x,
    act=paddle.activation.Linear(),
    bias_attr=paddle.attr.Param(
        initial_mean=0.,
        initial_std=0.01
    ),
    param_attr=paddle.attr.Param(
        initial_std=0.01,
        initial_mean=0.
    )
)

output = paddle.layer.last_seq(recurrent)

loss = paddle.layer.mse_cost(input=output, label=label)

parameters = paddle.parameters.create(loss)

optimizer = paddle.optimizer.RMSProp(
    regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
    learning_rate=0.001,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
)


trainer = paddle.trainer.SGD(
    cost=loss,
    parameters=parameters,
    update_equation=optimizer
)

feeding = {'x': 0, 'y': 1}

train_file_path = 'data/train.data'


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print ("\n pass %d, Batch: %d cost: %f, %s" % (event.pass_id, event.batch_id, event.cost, event.metrics))
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
            reader=paddle.batch(data_provider.data_reader(filepath), batch_size=128),
            feeding=feeding)
        print ("\nTest with Pass %d, %s" % (event.pass_id, result.metrics))


reader=data_provider.data_reader(train_file_path)
trainer.train(paddle.batch(reader=reader, batch_size=128), event_handler=event_handler, feeding=feeding)


