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

# __all___ = []

conv_file_path = 'dgk_shooter_min.conv'


def collect_conversations(filepath):

    def strip_to_sentences(line):
        line = line.split(' ')[1]
        line = line.rstrip('\n\r')
        line = line.replace('/', '')
        return line

    conversations = []
    print("Starting to collect conversations from file")
    with open(filepath, 'r') as f:
        lines = f.readlines()
        temp_conversation = []
        for line in lines:
            if line[0] == 'E':
                if len(temp_conversation):
                    conversations.append(temp_conversation)
                    temp_conversation = []
            elif line[0] == 'M':
                sentence = strip_to_sentences(line.decode('utf-8'))
                temp_conversation.append(sentence)
    print("Finishing collecting conversations count=%s" % len(conversations))
    return conversations


def generate_question_answers(conversations):

    questions = []
    answers = []

    for conv in conversations:
        if len(conv) % 2 != 0:
            conv = conv[:-1]

        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(conv[i])
            else:
                answers.append(conv[i])
    return questions, answers


def split_dataset(questions, answers, testset_size, trainset_size):
    train_questions = []
    train_answers = []
    test_questions = []
    test_answers = []

    test_index = random.sample([i for i in range(len(questions))], testset_size)

    indexes = range(len(questions))

    random.shuffle(indexes)

    for i in indexes:
        if i in test_index:
            test_questions.append(questions[i])
            test_answers.append(answers[i])
        else:
            train_questions.append(questions[i])
            train_answers.append(answers[i])
    trainset = dict()
    trainset['questions'] = train_questions[:trainset_size]
    trainset['answers'] = train_answers[:trainset_size]

    testset = dict()
    testset['questions'] = test_questions
    testset['answers'] = test_answers

    return trainset, testset


PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD = '__PAD__'
GO = '__GO__'
EOS = '__EOS__'
UNK = '__UNK__'


def build_vocabulary(trainset, testset, vocabulary_size=10000):
    vocabulary = {}
    conversations = trainset['questions']
    conversations += trainset['answers']
    conversations += testset['questions']
    conversations += testset['answers']
    print("Building the dictionary")
    for line in conversations:
        tokens = [token for token in line.strip()]
        for token in tokens:
            if token in vocabulary:
                vocabulary[token] += 1
            else:
                vocabulary[token] = 1

    vocabulary_list = [PAD, GO, EOS, UNK] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    if len(vocabulary_list) > vocabulary_size:
        vocabulary_list = vocabulary_list[:vocabulary_size]
    print("Finishing building")
    print("Saving the dictionary")
    with open('dictionary', 'w') as f:
        for token in vocabulary_list:
            f.write(token + '\n')
    print("Finishing saving the dictionary")


def load_vocabulary(vocabulary_path):
    print("Loading Vocabulary")
    f = open(vocabulary_path, 'r')
    vocabulary_lines = f.readlines()
    f.close()
    vocabulary_list = [token.rstrip('\n') for token in vocabulary_lines]
    vocabulary_dict = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
    del vocabulary_lines
    del vocabulary_list
    print("Finishing loading Vocabulary")
    return vocabulary_dict


def convert_sentence_to_vector(sentences, vocabulary, outputfile):

    output_sentences = []
    print("Converting sentences to vectors")
    for sentence in sentences:
        line_vector = []
        for token in sentence.strip():
            line_vector.append(vocabulary.get(token, UNK_ID))
        output_sentences.append(line_vector)

    with open(outputfile, 'w') as openfile:
        for sentence_vector in output_sentences:
            openfile.write(','.join([str(x) for x in sentence_vector]) + '\n')

    print("Finishing converting sentences to vectors")
    print("Saving to file: %s" % outputfile)









def test():
    print("testing")
    #convs = collect_conversations(conv_file_path)
    #print(convs[2][1])


test()




