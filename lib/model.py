#!/usr/bin/env python
# -*- coding: windows-1250 -*-

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed, RepeatVector
from keras.utils import multi_gpu_model
#from sklearn.model_selection import train_test_split
from keras import optimizers

from math import log
from .textUtil import TextUtil


#from keras.layers.wrappers import TimeDistributed, Bidirectional

import pickle
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical

from keras.models import Sequential

import time
import numpy as np


class ChatbotModel:
    def __init__(self):
        pass



    def defineModel(self, n_input, n_output, n_units):
        # define training encoder

        dropout = .20
        encoder_inputs = Input(shape=(None, n_input), name='Encoder-Input')
        encoder = LSTM(n_units, dropout=dropout, return_state=True, name='Encoder-LSTM')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        # define training decoder
        decoder_inputs = Input(shape=(None, n_output), name='Decoder-Input')
        decoder_lstm = LSTM(n_units, dropout=dropout, return_sequences=True, return_state=True, name='Decoder-LSTM')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax',name='Decoder-Dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Full-model')


        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        plot_model(model, to_file='oneHotModel.png', show_shapes=True)
        return model, encoder_model, decoder_model



    def defineEmbeddingModel(self,src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):


        model = Sequential()

        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
        model.add(LSTM(n_units))
        model.add(RepeatVector(tar_timesteps))
        model.add(LSTM(n_units, return_sequences=True))
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        # summarize defined model
        model.summary()
        #plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def seq2seq(self,latent_dim,doc_length, num_encoder_tokens, num_decoder_tokens):


        encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')
        x = Embedding(num_encoder_tokens,
                      latent_dim,
                      name='Body-Word-Embedding',
                      mask_zero=False)(encoder_inputs)
        x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

        dropout = .20



        _, state_h = GRU(latent_dim, dropout=dropout, return_state=True, name='Encoder-Last-GRU')(x)

        # create encoder model
        encoder_model = Model(inputs=encoder_inputs,
                              outputs=state_h,
                              name='Encoder-Model')

        seq2seq_encoder_out = encoder_model(encoder_inputs)
        # decoder model
        decoder_inputs = Input(shape=(None,), name='Decoder-Input')

        dec_emb = Embedding(num_decoder_tokens,
                            latent_dim,
                            name='Decoder-Word-Embedding',
                            mask_zero=False)(decoder_inputs)

        dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

        decoder_gru = GRU(latent_dim, dropout=dropout,return_state=True,return_sequences=True,name='Decoder-GRU')
        #decoder_gru = LSTM(latent_dim, dropout=dropout, return_state=True, return_sequences=True, name='Decoder-LSTM')

        decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)


        x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

        decoder_dense = Dense(num_decoder_tokens,
                              activation='softmax',
                              name='Final-Output-Dense')

        decoder_outputs = decoder_dense(x)


        seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        #paralell_model = multi_gpu_model(seq2seq_Model,gpus=2)

        seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001),loss='sparse_categorical_crossentropy')

        #seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

        plot_model(seq2seq_Model,'seq2seq.png',show_shapes=True,show_layer_names=True)

        print(seq2seq_Model.summary())

        return seq2seq_Model

    def getPredictions(self,encoder_vector, decoder_model, state_value):
        preds, st = decoder_model.predict([state_value, encoder_vector])

        indexProbs = []

        preds = preds.reshape(-1)


        for i in range(1, len(preds)-1):
            #-log(preds[i])
            candidate = [[i], preds[i]]
            indexProbs.append(candidate)


        return indexProbs, st

    # defines beam search
    def beamSearch(self,parentWord,actualWord,max_len, encoder_vector,decoder_model,K,level):
        state_value = np.array(actualWord).reshape(1, 1)
        # predict probs

        indexProbs,st = self.getPredictions(encoder_vector,decoder_model,state_value)

        # get K best
        KbestElements = sorted(indexProbs, key=lambda tup: tup[1],reverse=True)[0:K]

        activeCandidates,st  = self.beamSearchStep(KbestElements,st,decoder_model,K)

        for active in activeCandidates:
            sentence = ''
            for index in active[0]:
                sentence = sentence + self.int_to_word_input[index]+' '
            print(sentence,active[1])



    def beamSearchStep(self, activeCandidates,st, decoder_model,K):

        print('called beam search step',len(activeCandidates))

        # broaden active candidates sequence and update probabilities

        newActiveCandidates = []

        for candidate in activeCandidates:

            # stopping condition for recurse
            if(len(candidate[0]) == 15):
                return activeCandidates, st
            lastWord = candidate[0][len(candidate[0])-1]
            state_value = np.array(lastWord).reshape(1, 1)

            if(len(candidate) == 2):
                # beginning of the beam search
                encoder_vector = st
            else:
                encoder_vector = candidate[2]

            children, st = self.getPredictions(encoder_vector, decoder_model, state_value)

            childCandidates = []


            for child in children:
                # compute parent with everything
                combinedProb = candidate[1] * child[1]
                sequence = child[0][0]



                childCandidate = [sequence, combinedProb,st]
                childCandidates.append(childCandidate)
                sequence = candidate[0]

            # get only top K for each parent
            childCandidates = sorted(childCandidates, key=lambda tup: tup[1],reverse=True)[:K]

            seq = ' '.join(str(x) for x in candidate[0])

            for childCandidate in childCandidates:

                childCandidate[0] = seq+' '+str(childCandidate[0])
                childCandidate[0] = [int(x) for x in childCandidate[0].split(' ')]
                newActiveCandidates.append(childCandidate)


            childCandidates = []

        newActiveCandidates = sorted(newActiveCandidates, key=lambda tup: tup[1], reverse=True)[0:K]

        newActiveCandidates, st = self.beamSearchStep(newActiveCandidates,st,decoder_model,K)

        return newActiveCandidates,st



    def generateSentence(self,
                             raw_input_text,encoder_model,decoder_model,word_to_int_input, int_to_word_input,
                             max_len_answer=None, ):
        """
        Use the seq2seq model to generate an answer given the question
        Inputs
        ------
        raw_input: str
            The question text as an input string
        max_len_answer: int (optional)
            The maximum length of the asnwer the model will generate
        """

        self.int_to_word_input = int_to_word_input
        self.word_to_int_input = word_to_int_input


        # max_len_title
        # tokenized source text
        # encoder_vector


        # get the encoder's features for the decoder

        encoder_vector = encoder_model.predict(raw_input_text)


        startingWord = word_to_int_input['_start_']

        # use beam search
        self.beamSearch(None, startingWord, max_len_answer, encoder_vector, decoder_model, 5, 0)


        # use greedy search - if the greedy decoder has to be used then following section can be uncommented

        state_value = np.array(startingWord).reshape(1, 1)

        decoded_sentence = []

        '''

        stop_condition = False
        while not stop_condition:
            preds, st = decoder_model.predict([state_value, encoder_vector])

            probs = preds.reshape(-1)
            indexProb = []
            for i in range(0,len(probs)-1):
                # format : index, probability
                element = [i,probs[i]]
                indexProb.append(element)

            # select index - sort by prob in descending order and select first element
            pred_idx = sorted(indexProb, key=lambda tup: tup[1], reverse=True)[0][0]
            # empty index prob for next iteration
            indexProb = []


            # retrieve word from index prediction
            pred_word_str = self.int_to_word_input[pred_idx]

            print(pred_word_str)


            if len(decoded_sentence) >= max_len_answer:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            encoder_vector = st
            state_value = np.array(pred_idx).reshape(1, 1)
        '''

        return ' '.join(decoded_sentence)

    def extract_encoder_model(self,model):
        """
        Extract the encoder from the original Sequence to Sequence Model.
        Returns a keras model object that has one input (body of issue) and one
        output (encoding of issue, which is the last hidden state).
        Input:
        -----
        model: keras model object
        Returns:
        -----
        keras model object
        """
        encoder_model = model.get_layer('Encoder-Model')
        return encoder_model

    def extract_decoder_model(self,model):
        """
        Extract the decoder from the original model.
        Inputs:
        ------
        model: keras model object
        Returns:
        -------
        A Keras model object with the following inputs and outputs:
        Inputs of Keras Model That Is Returned:
        1: the embedding index for the last predicted word or the <Start> indicator
        2: the last hidden state, or in the case of the first word the hidden state
           from the encoder
        Outputs of Keras Model That Is Returned:
        1.  Prediction (class probabilities) for the next word
        2.  The hidden state of the decoder, to be fed back into the decoder at the
            next time step

        """

        latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]

        # Reconstruct the input into the decoder
        decoder_inputs = model.get_layer('Decoder-Input').input
        dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
        dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)


        gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')



        gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn,
                                                                 gru_inference_state_input])

        # Reconstruct dense layers
        dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
        dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
        decoder_model = Model([decoder_inputs, gru_inference_state_input],
                              [dense_out, gru_state_out])
        return decoder_model




    def decode_sequence(self,input_seq,words_len,vocabulary):
        # Encode the input as state vectors.

        num_decoder_tokens = words_len

        states_value = self.encoder_model.predict(input_seq)
        print(len(states_value))


        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 2] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        max_decoder_seq_length = 15
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])


            sampled_word = vocabulary[sampled_token_index]


            decoded_sentence.append(sampled_word)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def invert(self,seq):
        strings = list()
        for pattern in seq:

            min = pattern.min()
            max = pattern.max()

            indexProbs = []

            for i in range(0,len(pattern)-1):
                indexProb = [i,pattern[i]]
                indexProbs.append(indexProb)
            max = sorted(indexProbs, key=lambda tup: tup[1], reverse=True)[1][0]

            indexProbs = []


            #probs = pattern.argsort()[-3:]

            #max = probs[::1][0]

            string = self.int_to_word_input[max]


            if (string != "padd"):
                strings.append(string)
                print(string)
            else:
                return ' '.join(strings)

    def predict_sequence(self,source, n_steps, word_to_int_input, int_to_word_input):

        self.int_to_word_input = int_to_word_input
        # encode
        state = self.encoder_model.predict(source)
        # start of sequence input
        word = word_to_int_input['_start_']

        one_hot = to_categorical(word,len(word_to_int_input))

        firstWord = array([[word]])
        target_seq = array(TextUtil.one_hot_encode(firstWord, len(word_to_int_input)))


        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char

            yhat, h, c = self.decoder_model.predict([target_seq] + state)

            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]


            # update target sequence
            target_seq = yhat


        return self.invert(array(output))
