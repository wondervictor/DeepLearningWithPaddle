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
import imdb




def model(x):

    embedding_data = paddle.layer.embedding(
        input=x,
        size=512
    )

    bilstm_1 = paddle.networks.bidirectional_lstm(
        input=embedding_data,
        size=512,
        return_seq=True
    )

    fc_1 = paddle.layer.fc(
        input=bilstm_1,
        size=1024,
        act=paddle.activation.Relu()
    )

    bilstm_2 = paddle.networks.bidirectional_lstm(
        input=fc_1,
        size=512,
        return_seq=True
    )

    fc_2 = paddle.layer.fc(
        input=bilstm_2,
        size=1024,
        act=paddle.activation.Relu()
    )

    avg_pool = paddle.layer.pooling(
        input=fc_2,
        pooling_type=paddle.pooling.Avg()
    )

    output = paddle.layer.fc(
        input=avg_pool,
        size=2,
        act=paddle.activation.Softmax()
    )

    return output


def train():

    paddle.init(use_gpu=False, trainer_count=1)

    data = paddle.layer.data(
        name='data',
        type=paddle.data_type.integer_value_sequence(102100)
    )

    label = paddle.layer.data(
        name='label',
        type=paddle.data_type.integer_value(2)
    )

    output = model(data)

    loss = paddle.layer.classification_cost(
        input=output,
        label=label
    )

    parameters = paddle.parameters.create(loss)
    print(parameters.keys())

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001
    )

    trainer = paddle.trainer.SGD(
        parameters=parameters,
        update_equation=optimizer,
        cost=loss
    )
    path = 'data/imdb.pkl'
    dataset = imdb.Imdb(path)
    train_data_reader = dataset.create_reader('train')
    test_data_reader = dataset.create_reader('test')

    feeding = {
        'data': 0,
        'label': 1
    }

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 5 == 0:
                class_error_rate = event.metrics['classification_error_evaluator']
                print ("\npass %d, Batch: %d cost: %f error: %s" % (event.pass_id, event.batch_id, event.cost, class_error_rate))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            result = trainer.test(
                reader=paddle.batch(test_data_reader, batch_size=32),
                feeding=feeding)
            class_error_rate = result.metrics['classification_error_evaluator']
            print ("\nTest with Pass %d, cost: %s error: %f" % (event.pass_id, result.cost,class_error_rate))

    trainer.train(
        reader=paddle.batch(
            train_data_reader,
            batch_size=32
        ),
        event_handler=event_handler,
        num_passes=10,
        feeding=feeding
    )


if __name__ == '__main__':
    train()