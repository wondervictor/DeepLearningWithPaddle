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

import trainer
import data_generator
import matplotlib.pyplot as plt

data_generator.generator_sine_wave_data(99.43, 111, 10, 'valid.data')


data = []
with open('data/valid.data', 'r') as f:
    lines = f.readlines()
    for line in lines:
        element = map(float, line.rstrip('\n\r').split(','))
        data.append(element)

label = [p[-1] for p in data]

x = [[p[:-1]] for p in data]

s = trainer.train(x, 'output/model.tar.gz', True)
predict = [x[0] for x in s]


print("[%s]"% (' '.join(['%s' % x for x in label])))

print("[%s]"% (' '.join(['%s' % x for x in predict])))

# print("[")
# for i in range(len(s)):
#     print(label[i])
# print("]")
#
# print("[")
# for i in range(len(s)):
#     print(predict[i])
# print("]")

def mse(x, y):
    sum_val = 0.0
    for i in range(len(x)):
        sum_val += (x[i]-y[i])**2
    return sum_val/len(x)

print("MSE: %s" % (mse(label, predict)))

# 0.357120318785
# plt.figure(1)
#
# x = range(0, len(label))
#
# plt.plot(x, predict)
# plt.plot(x, label)
# plt.show()
