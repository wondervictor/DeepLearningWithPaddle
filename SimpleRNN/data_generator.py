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

import math


"""
Sine Wave Data Generator
"""

def generator_sine_wave_data(first_num, n, timestep, filename):

    def sine_f(x):
        x = x + first_num
        return 4*math.sin(3*x) + 2.5*math.cos(6*x)+10*math.sin(x) + 10.3

    data = []
    for i in range(0, n-timestep-1):
        x = [sine_f(p) for p in range(i, i+timestep)]
        y = sine_f(i+timestep)
        data.append((x, y))

    with open('data/%s' % filename, 'w+') as openFile:
        for data_pair in data:
            line = ','.join(['%s' % x for x in data_pair[0]])
            line += ',%s\n' % data_pair[1]
            openFile.write(line)



if __name__ == '__main__':

    generator_sine_wave_data(3.2, 10011, 10, 'train.data')

    generator_sine_wave_data(5.8, 1011, 10, 'test.data')



