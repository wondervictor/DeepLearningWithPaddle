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
import random
import numpy as np
from captcha.image import ImageCaptcha
import cv2
import matplotlib.pyplot as plt


def generate_num():
    nums = random.randint(1000, 99999999)
    code = str(nums)
    label = [int(j) for j in code]
    return code, label


def generate_image(code, captcha):
    img = captcha.generate(code)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 32))
    img = np.multiply(img, 1 / 255.0)
    img = np.reshape(img, [-1])
    return img


def generate_data(training_size, testing_size):
    captcha = ImageCaptcha(fonts=['OpenSans-Regular.ttf'])
    train_data = []
    train_label = []
    for i in range(training_size):
        print('generate train image: %s/%s' % (i+1, training_size))
        code, label = generate_num()
        img = generate_image(code, captcha)
        train_data.append(img)
        train_label.append(label)
    train_label = np.array(train_label)
    train_data = np.array(train_data)
    np.save('train_data', train_data)
    np.save('train_label', train_label)

    del train_label
    del train_data

    test_label = []
    test_data = []
    for i in range(testing_size):
        print('generate test image: %s/%s' % (i+1, testing_size))
        code, label = generate_num()
        img = generate_image(code, captcha)
        test_data.append(img)
        test_label.append(label)
    test_label = np.array(test_label)
    test_data = np.array(test_data)
    np.save('test_data', test_data)
    np.save('test_label', test_label)


def _test():
    data = np.load('train_data.npy')
    label = np.load('train_label.npy')

    img = np.reshape(data[100], [32, 100])
    print(label[100])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    #generate_data(8000, 1000)
    _test()
