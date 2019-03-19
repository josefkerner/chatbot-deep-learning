import keras

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model


def get_context_vec(context_mat, att_weigts):
    att_weigts_rep = K.expand_dims(att_weigts, 2)
    att_weigts_rep = K.repeat_elements(att_weigts_rep, context_mat.shape[2], 2)
    return K.sum(att_weigts_rep * context_mat, axis=1)


def attend(key_vec, context_mat, contextMatTimeSteps, w1, w2):
    key_rep = K.repeat(key_vec, contextMatTimeSteps)
    concated = K.concatenate([key_rep, context_mat], axis=-1)
    concated_r = K.reshape(concated, (-1, concated.shape[-1]))
    att_energies = K.dot((K.dot(concated_r, w1)), w2)
    att_energies = K.relu(K.reshape(att_energies, (-1, contextMatTimeSteps)))
    att_weigts = K.softmax(att_energies)

    return get_context_vec(context_mat, att_weigts), att_weigts


# the input is the  [ input , context_matrix ]

class AttentionDecoder(Layer):

    def __init__(self, rnn_cell, **kwargs):

        self.output_dim = rnn_cell.state_size[0]
        self.rnn_cell = rnn_cell
        super(AttentionDecoder, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list
        assert len(input_shape) == 2

        self.att_kernel = self.add_weight(name='att_kernel_1',
                                          shape=(self.output_dim + input_shape[1][2], input_shape[1][2]),
                                          initializer='uniform',
                                          trainable=True)

        self.att_kernel_2 = self.add_weight(name='att_kernel_2',
                                            shape=(input_shape[1][2], 1),
                                            initializer='uniform',
                                            trainable=True)

        step_input_shape = (
        input_shape[0][0], input_shape[0][2] + input_shape[1][2])  # batch_size , in_dim + contextVecDim
        self.rnn_cell.build(step_input_shape)

        self._trainable_weights += (self.rnn_cell.trainable_weights)
        self._non_trainable_weights += (self.rnn_cell.non_trainable_weights)

        self.contextMatTimeSteps = input_shape[1][1]

        super(AttentionDecoder, self).build(input_shape)

    def get_initial_state(self, inputs):

        initial_state = K.zeros_like(inputs)
        initial_state = K.sum(initial_state, axis=(1, 2))
        initial_state = K.expand_dims(initial_state)
        if hasattr(self.rnn_cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim]) for dim in self.rnn_cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.rnn_cell.state_size])]

    def call(self, input):
        inputs, context_mat = input

        def step(inputs, states):
            hid = states[0]
            ctx_vec, att_weigts = attend(hid, context_mat, self.contextMatTimeSteps, self.att_kernel, self.att_kernel_2)
            rnn_inp = K.concatenate((inputs, ctx_vec), axis=1)
            return self.rnn_cell.call(rnn_inp, states)

        timesteps = inputs.shape[1]

        initial_state = self.get_initial_state(inputs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             input_length=timesteps)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def predict(sent,word_to_int_input,int_to_word_input,model):

    ret = ""

    print(sent)

    m_input = [np.zeros((1, 100)), np.zeros((1, 100))]
    for i, w in enumerate(sent):
        m_input[0][0, i] = w
    m_input[1][0, 0] = word_to_int_input['_start_']

    for w_i in range(1, 35):
        out = model.predict(m_input)
        out_w_i = out[0][w_i - 1]



        probs = out_w_i.reshape(-1)
        indexProb = []
        for i in range(0, len(probs) - 1):
            # format : index, probability
            element = [i, probs[i]]
            indexProb.append(element)

        # select index - sort by prob in descending order and select first element
        out_w_i = sorted(indexProb, key=lambda tup: tup[1], reverse=True)[1][0]


        if out_w_i == 0:
            continue

        ret += int_to_word_input[out_w_i] + " "
        m_input[1][0, w_i] = out_w_i

    return ret


def getModel(enc_seq_length=35, enc_vocab_size=40005, dec_seq_length=35, dec_vocab_size=40005):
    inp = Input((enc_seq_length,))

    dropout = 0.20

    latent_dim = 256

    imp_x = Embedding(enc_vocab_size, latent_dim,name='Encoder-Input')(inp)

    #ctxmat = LSTM(256, dropout=dropout, return_sequences=True)(imp_x)
    ctxmat = Bidirectional(LSTM(latent_dim, dropout=dropout, return_sequences=True,name='Encoder-LSTM'),name='Bidirectional-LSTM')(imp_x)

    inp_cond = Input((dec_seq_length,),name='Decoder-Input')
    inp_cond_x = Embedding(dec_vocab_size, latent_dim, name='Decoder-Input-Embedding')(inp_cond)

    decoded = AttentionDecoder(LSTMCell(latent_dim,name='Decoder-LSTM'),name='Attention-Decoder')([inp_cond_x, ctxmat])
    decoded = TimeDistributed(Dense(dec_vocab_size, activation='softmax'),name='Decoder-Output-Layer')(decoded)

    model = Model([inp, inp_cond], decoded)

    #paralell_model = multi_gpu_model(model,gpus=2)
    optimizer = optimizers.Nadam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    #plot_model(model, to_file='AttentionModel.png', show_shapes=True)

    print(model.summary())

    return model