import os
import gzip
import math
import numpy as np
import pickle
import paddle.v2 as paddle
from model import predict_caption_net


def translate(output, id2word):
    probs = output[0][0]
    sentence = output[1]

    result = []
    tmp = []
    idx = 0

    def remove_eos(word):
        if word == 'EOS':
            return False
        return True

    for word in sentence:
        if word == -1:
            p = [id2word[x] for x in tmp]
            p = p[1:]
            p = filter(remove_eos, p)
            p = ' '.join(p)
            result.append((probs[idx], p))
            idx += 1
            tmp = []
        else:
            tmp.append(word)

    return result


def generate(image_path):

    paddle.init(use_gpu=False, trainer_count=1)

    DATA_DIM = 224 * 224 * 3
    DICT_DIM = 4529
    image = paddle.layer.data(name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    output = predict_caption_net(image, DICT_DIM)

    parameters = paddle.parameters.Parameters.from_tar(gzip.open('params/params_pass_5.tar.gz'))

    with open('dict/id2word.pkl', 'rb') as f:
        id2word = pickle.load(f)

    img = paddle.image.load_and_transform(
        filename=image_path,
        resize_size=224,
        crop_size=224,
        is_train=True,
        is_color=True
    ).flatten().astype('float32')

    inferer = paddle.inference.Inference(output_layer=output, parameters=parameters)
    result = inferer.infer(input=[(img,)], field=["prob", "id"])

    result = translate(result, id2word)
    print("[Image]: {}".format(image_path))
    for p in result:
        print("[Prob: {}] {}".format(p[0], p[1]))


if __name__ == '__main__':

    generate('/Users/Vic/Downloads/Flicker8k_Dataset/23445819_3a458716c1.jpg')
    # generate('/Users/Vic/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/ImageCaption/109202756_b97fcdc62c.jpg')
    # generate('/Users/Vic/Downloads/Flicker8k_Dataset/10815824_2997e03d76.jpg')



