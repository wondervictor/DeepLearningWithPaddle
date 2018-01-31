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

import matplotlib.pyplot as plt
import numpy

train_cost = []
test_cost = []
with open('log', 'r') as f:
    lines = f.readlines()

for line in lines:
    if line[0] == 'T':
        # test
        elements = line.rstrip('\n\r').split(':')
        cost = float(elements[1])
        test_cost.append(cost)
    elif line[0] == 'p':
        # pass
        elements = line.rstrip('\n\r').split(':')

        cost = float(elements[2])
        train_cost.append(cost)
    else:
        continue

plt.subplot(2,1,1)
plt.plot(train_cost, 'r')
plt.title('Training Loss')
plt.xlabel('iters')
plt.ylabel('loss')
plt.subplot(2,1,2)
plt.xlabel('iters')
plt.ylabel('loss')
plt.title('Testing Loss')
plt.plot(test_cost, 'g')
plt.show()