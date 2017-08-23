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

import paddle.v2 as paddle

learning_rate = 0.01
dict_dim = 10


def seq2seq(encoder_size, decoder_size, is_train):
    # input
    input_sentence_data = paddle.layer.data(
        name='input_sentence',
        type=paddle.data_type.integer_value_sequence(dict_dim)
    )

    input_data_embedding = paddle.layer.embedding(
        input=input_sentence_data,
        size=512
    )

    # encoder
    encoder_lstm = paddle.networks.bidirectional_lstm(
        input=input_data_embedding,
        size=encoder_size,
        return_seq=True
    )

    last_encoder = paddle.layer.last_seq(
        input=encoder_lstm
    )

    def _decoder(encoded_vector, current_input):

        memory_boot_layer = paddle.layer.fc(
            input=last_encoder,
            size=decoder_size,
            act=paddle.activation.Tanh()
        )

        decoder_memory = paddle.layer.memory(
            size=decoder_size,
            boot_layer=memory_boot_layer
        )

        encoded_context = paddle.last_seq(input=encoded_vector)
        input_data = paddle.layer.fc(
            input=[encoded_context, current_input],
            size=decoder_size * 3
        )

        step = paddle.layer.gru_step(
            input=input_data,
            size=decoder_size,
            output_mem=decoder_memory
        )

        output = paddle.layer.fc(
            input=step,
            size=dict_dim,
            act=paddle.activation.Softmax()
        )
        return output

    if is_train:
        current_input_word = paddle.layer.data(
            input='output_word',
            type=paddle.data_type.integer_value_sequence(dict_dim)
        )

        current_word_embedding = paddle.layer.embedding(
            input=current_input_word,
            size=512,
            name='current_word_embedding'
        )

        group_inputs = [paddle.layer.StaticInput(input=encoder_lstm)]
        group_inputs.append(current_word_embedding)

        decoder = paddle.layer.recurrent_group(
            step=_decoder,
            input=group_inputs
        )

        label = paddle.layer.data(
            name='label_word',
            type=paddle.data_type.integer_value_sequence(dict_dim)
        )

        cost = paddle.layer.classification_cost(input=decoder, label=label)

        return cost

    else:
        current_word_embedding = paddle.layer.GeneratedInput(
            size=dict_dim,
            embedding_name='current_word_embedding',
            embedding_size=512
        )

        group_inputs = [paddle.layer.StaticInput(input=encoder_lstm)]
        group_inputs.append(current_word_embedding)

        result = paddle.layer.beam_search(
            input=group_inputs,
            step=_decoder,
            bos_id=0,
            eos_id=1,
            beam_size=1,
            max_length=20
        )

        return result

def train():










