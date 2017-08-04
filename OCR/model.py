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

# cnn layers
def model(x):

    relu = paddle.activation.Relu()
    conv_1 = paddle.networks.img_conv_group(
        input=x,
        num_channels=3,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[32, 32],
        conv_filter_size=5,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_2 = paddle.networks.img_conv_group(
        input=conv_1,
        num_channels=32,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_3 = paddle.networks.img_conv_group(
        input=conv_2,
        num_channels=64,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[128, 128],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    conv_4 = paddle.networks.img_conv_group(
        input=conv_3,
        num_channels=128,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[256, 256],
        conv_filter_size=3,
        conv_act=relu,
        conv_with_batchnorm=True,
        pool_type=paddle.pooling.Max()
    )

    # pool_5 = paddle.layer.img_pool(
    #     input=conv_4,
    #     stride=2,
    #     pool_size=2,
    #     pool_type=paddle.pooling.Max()
    # )

    flatten = paddle.layer.fc(
        input=conv_4,
        size=512,
        act=relu
    )

    fc_1 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_2 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_3 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    fc_4 = paddle.layer.fc(
        input=flatten,
        size=10,
        act=paddle.activation.Softmax()
    )

    return [fc_1, fc_2, fc_3, fc_4]


IMAGE_SIZE = 32*80*3
LABEL_SIZE = 10


def train():
    paddle.init(use_gpu=False, trainer_count=1)

    x = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(IMAGE_SIZE),
        height=32,
        width=80
    )

    label = []
    for i in range(4):
        label_tmp = paddle.layer.data(
            name='label_part_%s' % i,
            type=paddle.data_type.integer_value(10)
        )
        label.append(label_tmp)

    output = model(x)

    loss = []
    for i in range(4):
        loss_tmp = paddle.layer.classification_cost(
            input=output[i],
            label=label[i]
        )
        loss.append(loss_tmp)
    loss = paddle.layer.addto(
        input=loss,
        bias_attr=False,
        act=paddle.activation.Linear()
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

    feeding = {'image': 0,
               'label_part_0': 1,
               'label_part_1': 2,
               'label_part_2': 3,
               'label_part_3': 4,
               }

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                print ("\npass %d, Batch: %d cost: %f metrics: %s" % (event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('output/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            test_reader = data_reader.create_reader('test')
            result = trainer.test(
                reader=paddle.batch(test_reader, batch_size=128),
                feeding=feeding)
            class_error_rate = result.metrics['classification_error_evaluator']
            print ("\nTest with Pass %d, cost: %s ratio: %f" % (event.pass_id, result.cost,class_error_rate))

    train_reader = data_reader.create_reader('train')
    reader = paddle.batch(
        reader=train_reader,
        batch_size=128
    )
    trainer.train(
        reader=reader,
        feeding=feeding,
        num_passes=10,
        event_handler=event_handler
    )


def predict(test_samples, model_path):
    paddle.init(use_gpu=False, trainer_count=1)

    x = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(IMAGE_SIZE),
        height=32,
        width=80
    )

    with gzip.open(model_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    output = model(x)

    result = paddle.infer(
        input=test_samples,
        parameters=parameters,
        output_layer=output,
        feeding={'image': 0}
    )

    return result


def generate_numbers(result):
    nums = []
    for i in range(4):
        max_value = 0.0
        max_index = -1
        for j in range(10):
            if max_value < result[i][j]:
                max_value = result[i][j]
                max_index = j
        nums.append(max_index)
    return nums


def test():
    model_path = '/Users/vic/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/OCR/output/params_pass_9.tar.gz'
    data, label = data_reader.testset()
    test_samples =[[data[10]]] #[[x] for x in data[10:15]]  #[[x] for x in data[10:20]]
    s = predict(test_samples, model_path)
    print(s)
    print(label[10])
    print(generate_numbers(s))
    # for i in range(5):
    #     print('-------------')
    #     print(label[10+i])


if __name__ == '__main__':
    test()








