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
import data_reader


NUM_CLASS = 10
IMG_HEIGHT = 32
IMG_WIDTH = 100

relu = paddle.activation.Relu()


# CNN Layers
def cnn(image):

    conv_group_1 = paddle.networks.img_conv_group(
        input=image,
        num_channels=1,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[32, 32],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )
    # 16 x 50 x 32

    conv_group_2 = paddle.networks.img_conv_group(
        input=conv_group_1,
        num_channels=32,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )
    # 8 x 25 x 64

    conv_3 = paddle.layer.img_conv(
        input=conv_group_2,
        num_channels=64,
        num_filters=128,
        filter_size=3,
        act=relu,
        stride=1,
        padding=1
    )

    conv_4 = paddle.layer.img_conv(
        input=conv_3,
        num_channels=128,
        num_filters=128,
        filter_size=3,
        act=relu,
        stride=1,
        padding=1
    )

    pool_1 = paddle.layer.img_pool(
        input=conv_4,
        pool_size=1,
        pool_size_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )
    # 4 x 25 x 64

    conv_5 = paddle.layer.img_conv(
        input=pool_1,
        num_channels=128,
        num_filters=256,
        filter_size=3,
        act=relu,
        stride=1,
        padding=1
    )

    pool_2 = paddle.layer.img_pool(
        input=conv_5,
        pool_size=1,
        pool_size_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )

    conv_6 = paddle.layer.img_conv(
        input=pool_2,
        num_channels=256,
        num_filters=256,
        filter_size=3,
        act=relu,
        stride=1,
        padding=1
    )

    pool_3 = paddle.layer.img_pool(
        input=conv_6,
        pool_size=1,
        pool_size_y=2,
        stride=1,
        stride_y=2,
        pool_type=paddle.pooling.Max()
    )
    # 25 x 256
    return pool_3


def bidirection_rnn(x, size, act):
    lstm_fw = paddle.networks.simple_lstm(
        input=x,
        size=128,
        act=relu
    )
    lstm_bw = paddle.networks.simple_lstm(
        input=x,
        size=128,
        act=relu,
        reverse=True
    )
    res = paddle.layer.fc(
        input=paddle.layer.concat(
            input=[lstm_fw, lstm_bw]
        ),
        act=act,
        size=size
    )

    return res


# RNN Layers
def rnn(x):
    x = bidirection_rnn(x, NUM_CLASS+1, paddle.activation.Softmax())
    #x = bidirection_rnn(x, NUM_CLASS+1, paddle.activation.Softmax())
    return x


def feature_to_sequences(x):
    return paddle.layer.block_expand(
        input=x,
        num_channels=256,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=1
    )


def model(x):
    cnn_part = cnn(x)
    feature_sequence = feature_to_sequences(cnn_part)
    rnn_part = rnn(feature_sequence)
    return rnn_part


def train():

    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(IMG_HEIGHT*IMG_WIDTH),
        height=IMG_HEIGHT,
        width=IMG_WIDTH
    )

    output = model(image)

    label = paddle.layer.data(
        name='label',
        type=paddle.data_type.integer_value_sequence(NUM_CLASS)
    )

    loss = paddle.layer.ctc(
        input=output,
        label=label,
        size=NUM_CLASS+1
    )

    feeding = {
        'image': 0,
        'label': 1
    }
    # with gzip.open('output/params_pass_0.tar.gz', 'r') as f:
    parameters = paddle.parameters.create(loss)

    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        learning_rate=0.01,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4)
    )

    trainer = paddle.trainer.SGD(
        cost=loss,
        parameters=parameters,
        update_equation=optimizer
    )

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 5 == 0:
                print ("\npass %d, Batch: %d cost: %f" % (event.pass_id+1, event.batch_id+1, event.cost))
            else:
                sys.stdout.write('*')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            test_reader = data_reader.create_reader('test')
            result = trainer.test(
                reader=paddle.batch(test_reader, batch_size=128),
                feeding=feeding)
            print ("\nTest with Pass %d, cost: %s" % (event.pass_id+1, result.cost))

    train_reader = data_reader.create_reader('train')
    reader = paddle.batch(
        reader=train_reader,
        batch_size=128
    )

    trainer.train(
        reader=reader,
        feeding=feeding,
        num_passes=50,
        event_handler=event_handler
    )


def generate_sequence(x):
    sequence = []
    prob_sequence = []
    for j in range(25):
        prob = x[j]
        maxValue = 0.0
        maxIndex = -1
        for i in range(11):
            if maxValue < prob[i]:
                maxValue = prob[i]
                maxIndex = i
        if maxIndex == 10:
            maxIndex = '-'
        sequence.append(maxIndex)
        prob_sequence.append(maxValue)
    return sequence, prob_sequence


def test(x):
    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(IMG_HEIGHT*IMG_WIDTH),
        height=IMG_HEIGHT,
        width=IMG_WIDTH
    )

    with gzip.open('output/params_pass_1.tar.gz', 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    
    output = model(image)

    result = paddle.infer(
        parameters=parameters,
        output_layer=output,
        input=x,
        feeding={
            'image': 0
        }
    )

    return result


if __name__ == '__main__':

    #train()

    data, label = data_reader.test(1)

    data = [[data]]

    res = test(data)
    print(res)
    res, probs = generate_sequence(res)
    print(probs)
    print(res)








