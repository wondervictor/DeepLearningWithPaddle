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

from captcha.image import ImageCaptcha
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


def random_numbers():
    nums = random.randint(1000, 9999)
    return nums


def generate_image(num, captcha):
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (80, 32))
    img = np.multiply(img, 1 / 255.0)
    img = np.reshape(img, [-1])
    return img


def generate_datasets(train_size, test_size):
    captcha = ImageCaptcha(fonts=['OpenSans-Regular.ttf'])
    train_data = []
    train_label = []
    for i in range(train_size):
        print('generate train image: %s/%s' % (i+1, train_size))
        num = random_numbers()
        code = str(num)
        img = generate_image(code, captcha)
        train_data.append(img)
        label = [int(j) for j in code]
        train_label.append(label)
    train_label = np.array(train_label)
    train_data = np.array(train_data)
    np.save('train_data', train_data)
    np.save('train_label', train_label)

    del train_label
    del train_data

    test_label = []
    test_data = []
    for i in range(test_size):
        print('generate test image: %s/%s' % (i+1, train_size))
        num = random_numbers()
        code = str(num)
        img = generate_image(code, captcha)
        test_data.append(img)
        label = [int(j) for j in code]
        test_label.append(label)
    test_label = np.array(test_label)
    test_data = np.array(test_data)
    np.save('test_data', test_data)
    np.save('test_label', test_label)


def _test():
    data_path = 'test_data.npy'
    label_path = 'test_label.npy'
    data = np.load(data_path)
    label = np.load(label_path)
    img = data[3]
    img = np.reshape(img, [32, 80, 3])
    print(label[3])
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    _test()
