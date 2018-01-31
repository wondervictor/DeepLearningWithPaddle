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

import mnist_data
import numpy as np


def create_reader(filename, n):
    def reader():
        if filename == 'train':
            dataset = mnist_data.fetch_traingset()
        else:
            dataset = mnist_data.fetch_testingset()
        for i in range(n):
            data = np.array(dataset['images'][i])
            data = np.reshape(data, (28, 28))
            yield data, dataset['labels'][i]

    return reader

"""

dataset['images'][i][0:28], \
                    dataset['images'][i][28:56], \
                    dataset['images'][i][56:84], \
                    dataset['images'][i][84:112], \
                    dataset['images'][i][112:140], \
                    dataset['images'][i][140:168], \
                    dataset['images'][i][168:196], \
                    dataset['images'][i][196:224], \
                    dataset['images'][i][224:252], \
                    dataset['images'][i][252:280], \
                    dataset['images'][i][280:308], \
                    dataset['images'][i][308:336], \
                    dataset['images'][i][336:364], \
                    dataset['images'][i][364:392], \
                    dataset['images'][i][392:420], \
                    dataset['images'][i][420:448], \
                    dataset['images'][i][448:476], \
                    dataset['images'][i][476:504], \
                    dataset['images'][i][504:532], \
                    dataset['images'][i][532:560], \
                    dataset['images'][i][560:588], \
                    dataset['images'][i][588:616], \
                    dataset['images'][i][616:644], \
                    dataset['images'][i][644:672], \
                    dataset['images'][i][672:700], \
                    dataset['images'][i][700:728], \
                    dataset['images'][i][728:756], \
                    dataset['images'][i][756:784], \
                    dataset['labels'][i]
"""