#!/usr/bin/env python
# -*- coding: windows-1250 -*-

import re
from random import randint
import keras

import csv
from keras import optimizers
import sys

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import load_model

from keras.callbacks import ModelCheckpoint

import numpy as np
from lib import attentionModel
from lib.attentionModel import getModel, predict

import pickle

import os

#train ON CPU instead of GPU if this option is active

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.preprocessing import sequence
import time
from lib.model import ChatbotModel
from lib.dataPreprocessor import DataPreprocessor
from lib.textUtil import TextUtil


# garbage collection
import gc

# optimizers to be tried

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam


class Trainer:
    def __init__(self):



        self.prepareVocab = 1 # 1 - trigger vocab generation, 0 - do not trigger
        self.maxExamples = 5100 # number of examples used in the training
        self.maxLen = 100 # max length of question and answer (in words)

        # parseArgs --------------------

        # modelMode = train or infer
        # modelType = which model to train or infer
            # one-hot, seq2seq2, embedding, attention
        self.modelMode, self.modelType = self.parseArgs()

        print('you have selected mode : '+self.modelMode+' and model type : '+self.modelType)


        if(self.modelMode == 'infer'):
            if(self.modelType == "embedding"):
                self.inferEmbeddingModel();
            elif(self.modelType == "attention"):
                self.inferAttentionModel();
            elif(self.modelType == "seq2seq"):
                self.inferSeq2SeqModel();
            else:
                self.triggerInference();
        elif(self.modelMode == 'train'):
            self.triggerTrainining()
        else:
            print('bad model mode, select either train or infer')
            exit(1)

    def parseArgs(self):
        import optparse

        parser = optparse.OptionParser()

        parser.add_option('-m', '--mode',
                          action="store", dest="mode",
                          help="mode of the model - train or infer", default="train")

        parser.add_option('-t', '--type',
                          action="store", dest="type",
                          help="type of the model - which model to use", default="seq2seq")

        options, args = parser.parse_args()

        return options.mode, options.type


    def triggerInference(self):
        # one hot model inference
        sentence = 'dobrý den, neumím číst. co mám dělat?'

        sentence = DataPreprocessor('heureka_train.csv').processText(sentence)

        self.model = ChatbotModel();

        word_to_int_input = pickle.load(open("word_to_int_input.pickle","rb"))
        int_to_word_input = pickle.load(open("int_to_word_input.pickle", "rb"))

        input_data = [word_to_int_input[word] for word in sentence]
        input_data = np.array([input_data])

        input_data = sequence.pad_sequences(input_data, maxlen=100, padding='post')
        input_data = TextUtil.one_hot_encode(input_data, len(word_to_int_input))

        train, infenc, infdec = self.model.defineModel(len(word_to_int_input), len(word_to_int_input), 256)

        self.model.encoder_model.load_weights('model_enc.h5')
        self.model.decoder_model.load_weights('model_dec.h5')

        target = self.model.predict_sequence(input_data, 15, word_to_int_input, int_to_word_input)

        #response = self.model.decode_sequence(integer_encoded, len(word_to_int_input),int_to_word_input)

        print(target)

    def inferSeq2SeqModel(self):

        start = int(time.time() * 1000.0)

        self.model = ChatbotModel();

        dataPreprocessor = DataPreprocessor('heureka_train.csv')

        sentence = 'dobrý den, neumím číst. co mám dělat?'

        sentence = dataPreprocessor.processText(sentence)

        word_to_int_input = pickle.load(open("word_to_int_input.pickle", "rb"))
        int_to_word_input = pickle.load(open("int_to_word_input.pickle", "rb"))

        input_data = [word_to_int_input[word] for word in sentence]
        input_data = np.array([input_data])

        input_data = sequence.pad_sequences(input_data, maxlen=100, padding='pre')

        num_tokens = len(word_to_int_input)

        seq2seqModel = self.model.seq2seq(256,100,num_tokens,num_tokens)

        seq2seqModel.load_weights('seq2seq256-2250data.hdf5')

        encoder_model = self.model.extract_encoder_model(seq2seqModel)
        decoder_model = self.model.extract_decoder_model(seq2seqModel)

        decoded = self.model.generateSentence(input_data,encoder_model,decoder_model,word_to_int_input,int_to_word_input,20)

        print(decoded)
        end = int(time.time() * 1000.0)
        timeElapsed = end - start

        print('time elapsed in miliseconds', timeElapsed)

    def integerEncodeSentence(self,sentence):

        self.dataPreprocessor = DataPreprocessor('heureka_train.csv')

        # sentence = sentence.split(' ')
        sentence = self.dataPreprocessor.processText(sentence)
        print(sentence)
        # sentence = TextUtil.cleanText(sentence)
        word_to_int_input = pickle.load(open("word_to_int_input.pickle", "rb"))
        int_to_word_input = pickle.load(open("int_to_word_input.pickle", "rb"))

        input_data = [word_to_int_input[word] for word in sentence]
        input_data = np.array([input_data])

        input_data = sequence.pad_sequences(input_data, maxlen=self.maxLen, padding='post')

        encoded_length = len(word_to_int_input)

        return input_data, encoded_length, word_to_int_input, int_to_word_input

    def inferAttentionModel(self):
        sentence = 'tak mi notebook poslali opět do servisu, tak mi držte palce, dík.'
        sentence, encoded_length, word_to_int_input, int_to_word_input = self.integerEncodeSentence(sentence)


        vocab_size = encoded_length

        print('obtaining the attention model')
        model = getModel(enc_seq_length=self.maxLen, enc_vocab_size=vocab_size, dec_seq_length=self.maxLen, dec_vocab_size=vocab_size)

        model.load_weights('attention_model256-2.hdf5')

        # call predict method from attentionModel.py
        result = predict(sentence[0],word_to_int_input, int_to_word_input,model)
        print(result)

    def inferEmbeddingModel(self):

        input_data = self.integerEncodeSentence()

        source = input_data.reshape((1, input_data.shape[1]))

        model = load_model('model.h5')

        prediction = model.predict(source, verbose=0)[0]

        for vector in prediction:
            vector = np.argsort(vector)[::-1]
            print(vector[1])

        integers = [argmax(vector) for vector in prediction]

        print(integers)
        target = list()


    def triggerTrainining(self):
        self.dataPreprocessor = DataPreprocessor('heureka_train.csv')


        self.model = ChatbotModel()

        if(self.prepareVocab == 1):
            self.prepareVocabulary();

        self.prepareTrainingVectors()

        if(self.modelType == 'embedding'):
            self.trainEmbeddingModel()
        elif(self.modelType == "attention"):
            self.trainAttentionModel();
        elif(self.modelType == 'seq2seq'):
            self.trainSeq2seqModel();

        else:
            self.trainModel()


    def prepareVocabulary(self):

        # count unique words in the training data

        self.dataX = [] # questions in text
        self.dataY = [] # answers in text

        words = []
        allWords = []

        for pair in self.dataPreprocessor.cleanedData:
            self.dataX.append(pair[0])
            self.dataY.append(pair[1])

            for sentence in pair:

                for word in sentence:
                    if(word not in words):
                        words.append(word)
                    allWords.append(word)

        print('number of all words',len(allWords))

        word_to_int_input = dict((c, i + 2) for i, c in enumerate(words))
        int_to_word_input = dict((i + 2, c) for i, c in enumerate(words))
        word_to_int_input.update({"padd": 0})
        int_to_word_input.update({0: "padd"})
        word_to_int_input.update({"_": 1})
        int_to_word_input.update({1: "_"})

        save_word_features = open("word_to_int_input.pickle", "wb")
        pickle.dump(word_to_int_input, save_word_features)
        save_word_features.close()
        save_word_features = open("int_to_word_input.pickle", "wb")
        pickle.dump(int_to_word_input, save_word_features)
        save_word_features.close()

        self.encoded_length = len(word_to_int_input)

        print('Vocabulary size is: ',self.encoded_length)



    def prepareTrainingVectors(self):


        # number of timesteps for encoder and decoder
        n_in = self.maxLen
        n_out = self.maxLen

        X1,X2,Y,max_sequence_len = self.dataPreprocessor.integerEncode()

        X1 = X1[0:self.maxExamples]
        X2 = X2[0:self.maxExamples]
        Y = Y[0:self.maxExamples]

        print('training vectors sample: ')
        print(X1[0])
        print(X2[0])
        print(Y[0])



        gc.collect()

        # zero padding
        X1 = sequence.pad_sequences(X1, maxlen=n_in, padding='pre')




        X2 = sequence.pad_sequences(X2, maxlen=n_out, padding='post')
        Y = sequence.pad_sequences(Y, maxlen=n_out, padding='post')

        print('zero padding finished')

        if(self.modelType == 'embedding'):
            self.X = X1
            self.Y = self.dataPreprocessor.encode_output(Y, max_sequence_len)

        elif(self.modelType == 'attention'):
            # prepare data for attention model
            self.X1 = X1

            self.X2 = Y[:, :-1]

            self.Y = Y[:, 1:]

            self.X2 = sequence.pad_sequences(self.X2, maxlen=n_out, padding='post')

            self.Y = sequence.pad_sequences(self.Y, maxlen=n_out, padding='post')

            self.X = [self.X1,self.X2]

            #self.Y = keras.utils.to_categorical( self.Y , num_classes=self.dataPreprocessor.encoded_length)

            #self.Y = TextUtil.one_hot_encode(self.Y, self.dataPreprocessor.encoded_length)
        elif(self.modelType == 'seq2seq'):

            # for embedding model

            self.X1 = X1

            self.X2 = Y[:, :-1]

            self.Y = Y[:, 1:]

        else:

            # for one hot model
            start = int(time.time() * 1000.0)

            X2 = Y[:, :-1]

            Y = Y[:, 1:]


            self.X1 = TextUtil.one_hot_encode(X1, max_sequence_len)

            self.X2 = TextUtil.one_hot_encode(X2, max_sequence_len)

            self.Y = TextUtil.one_hot_encode(Y, max_sequence_len)

            end = int(time.time() * 1000.0)
            timeElapsed = end - start
            print('time needed for one hot encoding:',timeElapsed,'miliseconds')

        del(X1)
        del(X2)
        del(Y)
        gc.collect()

        print('prepared training vectors sample: ')
        print(self.X1[0])
        print(self.X2[0])
        print(self.Y[0])

        print('vectors preparation ended, beginning '+self.modelType+' model training..')

    def trainSeq2seqModel(self):
        from keras.callbacks import CSVLogger

        latent_dim = 256
        epochs = 1
        batch_size = 64
        doc_length = self.maxLen

        num_encoder_tokens = self.dataPreprocessor.encoded_length
        num_decoder_tokens = self.dataPreprocessor.encoded_length

        script_name_base = 'seq2seq'
        csv_logger = CSVLogger('{:}.log'.format(script_name_base))
        model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                           save_best_only=True, mode='min')

        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0.001,
                                           patience=3,
                                           verbose=0, mode='auto')

        seq2seq_Model = self.model.seq2seq(latent_dim,self.maxLen,num_encoder_tokens, num_decoder_tokens)

        history = seq2seq_Model.fit([self.X1, self.X2], np.expand_dims(self.Y, -1),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.25, callbacks=[csv_logger, model_checkpoint])

        score = seq2seq_Model.evaluate([self.X1[0:50],self.X2[0:50]],np.expand_dims(self.Y[0:50],-1))

        print(score)

    def trainModel(self):
        # define model
        train, infenc, infdec = self.model.defineModel(self.dataPreprocessor.encoded_length, self.dataPreprocessor.encoded_length, 256)

        optimizer = optimizers.Nadam(lr=0.001)

        train.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(train.summary())

        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0.01,
                                      patience=2,
                                      verbose=0, mode='auto')

        checkpoint = ModelCheckpoint('modelOneHot.h5', monitor='val_loss', save_best_only=True, mode='min')

        # train model
        train.fit([self.X1, self.X2], self.Y, batch_size=128,epochs=30, validation_split=0.25, callbacks=[checkpoint,es])

        infenc.save_weights("model_enc.h5")

        infdec.save_weights("model_dec.h5")

    def trainEmbeddingModel(self):
        input_vocab_length = self.dataPreprocessor.encoded_length
        answer_vocab_length = self.dataPreprocessor.encoded_length
        question_max_length = 100
        answer_max_length = 100
        model = self.model.defineEmbeddingModel(input_vocab_length, answer_vocab_length, question_max_length, answer_max_length, 256)

        checkpoint = ModelCheckpoint('modelEmbedding.h5', monitor='val_loss', save_best_only=True, mode='min')

        model.fit(self.X, self.Y, epochs=30, batch_size=64, validation_split=0.3, callbacks=[checkpoint])

    def trainAttentionModel(self):


        vocab_size = self.dataPreprocessor.encoded_length

        print('obtaining the attention model')

        model = getModel(enc_seq_length=self.maxLen,enc_vocab_size=vocab_size,dec_seq_length=self.maxLen,dec_vocab_size=vocab_size)



        self.X1_test = self.X1[3825:len(self.X1)-1]
        self.X2_test = self.X2[3825:len(self.X2)-1]
        self.Y_test = self.Y[3825:len(self.Y)-1]

        self.X1 = self.X1[0:3824]
        self.X2 = self.X2[0:3824]
        self.Y = self.Y[0:3824]

        tr_data = range(self.X1.shape[0])
        tr_data_test = range(self.X1_test.shape[0])




        def load_data(batchSize=32):
            while True:
                for i in range(0,len(tr_data)-batchSize,batchSize):
                    inds = tr_data[i: i+batchSize]

                    yield [ self.X1[inds], self.X2[inds]], TextUtil.one_hot_encode(self.Y[inds], self.dataPreprocessor.encoded_length)

        def load_data_test(batchSize=32):
            while True:
                for i in range(0,len(tr_data_test)-batchSize,batchSize):
                    inds = tr_data_test[i: i+batchSize]

                    yield [ self.X1_test[inds], self.X2_test[inds]], TextUtil.one_hot_encode(self.Y_test[inds], self.dataPreprocessor.encoded_length)

        tr_gen = load_data(batchSize=32)
        tr_test = load_data_test(batchSize=32)


        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0.01,
                                           patience=2,
                                           verbose=0, mode='auto')

        checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format('attention_model'),
                                           save_best_only=True, mode='min')


        #model.fit(self.X, self.Y, epochs=30, batch_size=128, validation_split=0.25, callbacks=[checkpoint,es])



        model.fit_generator(generator=tr_gen, validation_data=tr_test, use_multiprocessing=True,
                            workers=4, steps_per_epoch=50,
                            validation_steps=30, epochs=30, callbacks=[checkpoint])

        score = model.evaluate([self.X1_test, self.X2_test], TextUtil.one_hot_encode(self.Y_test, self.dataPreprocessor.encoded_length))

        print('evaluation value loss and accuracy is: ',score)

        model.save_weights('attentionModel' + ".h5")


        print('finished training')

Trainer()