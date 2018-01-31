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
import paddle.trainer_config_helpers as tch
import sys
import gzip
import data_provider as dp

import seq2seq as seq_model

dict_dim = 4469



def seq2seq_attention(
        source_dict_dim,
        target_dict_dim,
        encoder_size,
        decoder_size,
        embedding_size,
        is_train,
):
    # input sentence
    input_sentence = paddle.layer.data(
        name='input_sentence',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    # input embedding
    input_emb = paddle.layer.embedding(
        input=input_sentence,
        size=embedding_size,
        param_attr=paddle.attr.ParamAttr(name='input_embedding_param'))

    fwd_lstm = paddle.networks.simple_lstm(
        name='fwd_lstm_layer',
        input=input_emb,
        size=encoder_size
    )

    bwd_lstm = paddle.networks.simple_lstm(
        name='bwd_lstm_layer',
        input=input_emb,
        size=encoder_size,
        reverse=True
    )

    # encoder result
    encoded_vector = paddle.layer.concat(
        input=[fwd_lstm, bwd_lstm]
    )

    fwd_last = paddle.layer.last_seq(
        input=fwd_lstm
    )
    bwd_first = paddle.layer.first_seq(
        input=bwd_lstm
    )

    boot_vector = paddle.layer.fc(
        input=paddle.layer.concat(
            input=[fwd_last, bwd_first]
        ),
        size=decoder_size
    )

    # projection for encoded vector
    with paddle.layer.mixed(size=decoder_size) as encoded_proj:
        encoded_proj += paddle.layer.full_matrix_projection(
            input=encoded_vector)

    # GRU Decoder with Attention
    def gru_attend_decoder(encoded_vector, encoded_projection, current_word):

        decoder_mem = paddle.layer.memory(
            name='gru_decoder',
            size=decoder_size,
            boot_layer=boot_vector
        )

        attend_context = paddle.networks.simple_attention(
            encoded_sequence=encoded_vector,
            encoded_proj=encoded_projection,
            decoder_state=decoder_mem
        )

        with paddle.layer.mixed(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=attend_context
            )
            decoder_inputs += paddle.layer.full_matrix_projection(
                input=current_word
            )

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size
        )

        with paddle.layer.mixed(
                size=target_dict_dim,
                bias_attr=True,
                act=paddle.activation.Softmax()) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)

        return out

    group_input1 = paddle.layer.StaticInputV2(
        input=encoded_vector,
        is_seq=True
    )
    group_input2 = paddle.layer.StaticInputV2(
        input=encoded_proj,
        is_seq=True
    )
    group_inputs = [group_input1, group_input2]

    if is_train:

        target_word = paddle.layer.data(
            name='target_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim)
        )

        target_embedding = paddle.layer.embedding(
            input=target_word,
            size=embedding_size,
            param_attr=paddle.attr.ParamAttr(
                name='target_input_embedding_param'
            )
        )
        group_inputs.append(target_embedding)

        decoder = paddle.layer.recurrent_group(
            name='decoder_group',
            step=gru_attend_decoder,
            input=group_inputs)

        label = paddle.layer.data(
            name='label_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim)
        )

        cost = paddle.layer.classification_cost(
            input=decoder,
            label=label
        )

        return cost
    else:
        target_embedding = paddle.layer.GeneratedInputV2(
            size=target_dict_dim,
            embedding_name='target_input_embedding_param',
            embedding_size=embedding_size
        )
        group_inputs.append(target_embedding)

        result = paddle.layer.beam_search(
            name='decoder_group',
            step=gru_attend_decoder,
            input=group_inputs,
            bos_id=1,
            eos_id=2,
            beam_size=3,
            max_length=30
        )

        return result

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
    encoded_vector = paddle.networks.bidirectional_lstm(
        input=input_data_embedding,
        size=encoder_size,
        fwd_act=paddle.activation.Tanh(),
        fwd_gate_act=paddle.activation.Sigmoid(),
        bwd_act=paddle.activation.Tanh(),
        bwd_gate_act=paddle.activation.Sigmoid(),
        return_seq=True
    )

    last_encoder = paddle.layer.last_seq(
        input=encoded_vector
    )

    def _decoder(encoder_vector, current_input):

        memory_boot_layer = paddle.layer.fc(
            input=encoder_vector,
            size=decoder_size,
            act=paddle.activation.Tanh()
        )

        decoder_memory = paddle.layer.memory(
            size=decoder_size,
            boot_layer=memory_boot_layer,
            name='gru_decoder'
        )

        input_data = paddle.layer.fc(
            input=current_input,
            size=decoder_size * 3
        )

        step = paddle.layer.gru_step(
            name='gru_decoder',
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

    group_input1 = paddle.layer.StaticInputV2(input=last_encoder)

    if is_train:

        current_input_word = paddle.layer.data(
            name='output_word',
            type=paddle.data_type.integer_value_sequence(dict_dim)
        )

        current_word_embedding = paddle.layer.embedding(
            input=current_input_word,
            size=512,
            name='current_word_embedding'
        )

        group_inputs = [group_input1, current_word_embedding]

        decoder = paddle.layer.recurrent_group(
            step=_decoder,
            input=group_inputs,
            name='decoder_group'
        )

        label = paddle.layer.data(
            name='label_word',
            type=paddle.data_type.integer_value_sequence(dict_dim)
        )

        cost = paddle.layer.classification_cost(input=decoder, label=label)

        return cost

    else:
        current_word_embedding = paddle.layer.GeneratedInputV2(
            size=dict_dim,
            embedding_name='current_word_embedding',
            embedding_size=512
        )

        group_inputs = [group_input1, current_word_embedding]

        result = paddle.layer.beam_search(
            name='decoder_group',
            input=group_inputs,
            step=_decoder,
            bos_id=1,
            eos_id=2,
            beam_size=3,
            max_length=30
        )

        return result


def train():

    paddle.init(use_gpu=False, trainer_count=1)

    cost = seq2seq_attention(
        source_dict_dim=dict_dim,
        target_dict_dim=dict_dim,
        encoder_size=512,
        decoder_size=512,
        embedding_size=512,
        is_train=True
    )

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.RMSProp(
        learning_rate=0.0001,
    )

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=optimizer
    )

    feeding = {
        'input_sentence': 0,
        'target_word': 1,
        'label_word': 2
    }

    wmt_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size=dict_dim), buf_size=1024),
        batch_size=64
    )
    #train_reader = data_reader.create_reader(True)

    trainer.train(
        num_passes=10,
        event_handler=event_handler,
        reader=paddle.batch(
            reader=train_reader,
            batch_size=64
        ),
        feeding=feeding
    )


def test():

    input_seq = [1229,199,134,1349,5,31,7,13,20,2496,1143]

    paddle.init(use_gpu=False, trainer_count=1)
    output = seq2seq_attention(
        source_dict_dim=dict_dim,
        target_dict_dim=dict_dim,
        encoder_size=512,
        decoder_size=512,
        embedding_size=512,
        is_train=False
    )

    with gzip.open('output/params_pass_1.tar.gz', 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    result = paddle.infer(
        output_layer=output,
        input=[(input_seq,)],
        parameters=parameters,
        field=['prob', 'id']
    )

    print(result)


test()

#train()













