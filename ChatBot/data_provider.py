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


__all__ = ['create_reader']


def open_file(path):
    data = []
    with open(path, 'r') as ff:
        lines = ff.readlines()
        for line in lines:
            line = line.rstrip('\r\n')
            seq = map(int, line.split(','))
            data.append(seq)
    return data

GO_ID = 1
EOS_ID = 2

# def test():
#     answer_path = 'data/train_answers'
#     question_path = 'data/train_questions'
#
#     answers = open_file(answer_path)
#     questions = open_file(question_path)
#
# test()


def create_reader(is_train=True):

    def reader():
        if is_train:
            answer_path = 'data/train_answers'
            question_path = 'data/train_questions'
            size = 20000
        else:
            answer_path = 'data/test_answers'
            question_path = 'data/test_questions'
            size = 9000
        questions = open_file(question_path)
        answers = open_file(answer_path)
        for i in range(size):
            yield ([GO_ID]+questions[i]+[EOS_ID]), ([GO_ID]+answers[i]), (answers[i]+[EOS_ID])

    return reader




