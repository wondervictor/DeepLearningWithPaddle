# -*- coding: utf-8 -*-

import paddle.v2 as paddle


# ResNet
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False,
        param_attr=paddle.attr.Param(is_static=True)
    )
    return paddle.layer.batch_norm(input=tmp, act=active_type)


def shortcut(input, ch_out, stride):
    if input.num_filters != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0,
                             paddle.activation.Linear())
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, paddle.activation.Linear())
    return paddle.layer.addto(
        input=[short, conv2], act=paddle.activation.Relu())


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0,
                          paddle.activation.Linear())
    return paddle.layer.addto(
        input=[short, conv3], act=paddle.activation.Relu())


def layer_warp(block_func, input, ch_out, count, stride):
    conv = block_func(input, ch_out, stride)
    for i in range(1, count):
        conv = block_func(conv, ch_out, 1)
    return conv


def resnet_imagenet(input, depth=50):
    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(
        input, ch_in=3, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = paddle.layer.img_pool(input=conv1, pool_size=3, stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = paddle.layer.img_pool(
        input=res4, pool_size=7, stride=1, pool_type=paddle.pooling.Avg())
    output = paddle.layer.fc(pool2, act=paddle.activation.Softmax(), size=1000)
    return pool2


# Decoder
def decoder(features, target, dict_dim, max_length=30, beam_size=3,decoder_size=512, embed_size=512, label=None, is_train=True):

    encoded_features = paddle.layer.fc(
        input=features,
        size=512,
        act=paddle.activation.Relu(),
        name='encoded_features'
    )

    def decoder_step(encoded_vector, current_word):
        decoder_memory = paddle.layer.memory(
            name="gru_decoder",
            size=decoder_size,
            boot_layer=encoded_features
        )
        context = encoded_vector
        decoder_inputs = paddle.layer.fc(input=[context, current_word], size=decoder_size * 3)

        gru_step = paddle.layer.gru_step(
            name="gru_decoder",
            act=paddle.activation.Tanh(),
            input=decoder_inputs,
            output_mem=decoder_memory,
            size=decoder_size
        )

        out = paddle.layer.fc(
            size=dict_dim,
            bias_attr=True,
            act=paddle.activation.Softmax(),
            input=gru_step
        )

        return out

    group_inputs = [paddle.layer.StaticInput(input=encoded_features)]
    if is_train:
        # training
        target_embed = paddle.layer.embedding(target, size=embed_size, name='embedding')
        group_inputs.append(target_embed)

        decoder_output = paddle.layer.recurrent_group(
            name='gru_group_decoder',
            step=decoder_step,
            input=group_inputs
        )
        cost = paddle.layer.classification_cost(input=decoder_output, label=label)
        return cost
    else:
        # generating
        target_embed = paddle.layer.GeneratedInput(
            size=dict_dim,
            embedding_name='embedding',
            embedding_size=embed_size
        )
        group_inputs.append(target_embed)

        beam_gen = paddle.layer.beam_search(
            name='gru_group_decoder',
            step=decoder_step,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen


# train model
def train_caption_net(input_images, target, label, dict_dim):
    encoder_ = resnet_imagenet(input_images)
    cost = decoder(features=encoder_, target=target, label=label, dict_dim=dict_dim)

    return cost


def predict_caption_net(input_images, target, dict_dim):
    encoder_ = resnet_imagenet(input_images)
    word = decoder(features=encoder_, target=target, dict_dim=dict_dim, is_train=False)
    return word

