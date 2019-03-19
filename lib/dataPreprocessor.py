import csv
import pickle
from numpy import array
from keras.utils import to_categorical
from lib.textUtil import TextUtil
import sys

class DataPreprocessor:
    def __init__(self, file):
        self.maxQuestionLen = 100
        self.maxAnswerLen = 100
        self.cleanedData = self.cleanData(file)





        self.X, self.Y = self.getTrainPairs();

    def integerEncode(self):

        self.word_to_int_input = pickle.load(open("word_to_int_input.pickle", "rb"))
        self.int_to_word_input = pickle.load(open("int_to_word_input.pickle", "rb"))

        self.encoded_length = len(self.word_to_int_input)


        X1 = []  # questions in integers
        X2 = []  # questions in integers
        Y = []


        # integer encoding
        for sentence in self.dataX:

            sentence = [word for word in sentence]



            X1.append([self.word_to_int_input[word] for word in sentence])
        for sentence in self.dataY:
            #print(sentence)
            #exit(0)
            #sentence = str(sentence).split(' ')
            #sentence = [word for word in sentence if word.isalpha()]

            Y.append([self.word_to_int_input[word] for word in sentence])

        # input for decoder - target sentence minus one last word
        for sentence in self.dataY:
            #sentence = str(sentence).split(' ')
            #sentence = [word for word in sentence if word.isalpha()]

            # sentence = sentence[0:len(sentence) - 1]

            X2.append([self.word_to_int_input[word] for word in sentence])

        print('X1 size',sys.getsizeof(X1))
        print('X2 size',sys.getsizeof(X2))
        print('Y size',sys.getsizeof(Y))



        return X1,X2,Y,self.encoded_length

    def encode_output(self,sequences, vocab_size):
        ylist = list()

        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

    def getTrainPairs(self):

        self.dataX = []  # questions in text
        self.dataY = []  # answers in text

        for pair in self.cleanedData:
            self.dataX.append(pair[0])
            self.dataY.append(pair[1])

        return self.dataX,self.dataY

    def cleanData(self, fileName):
        cleaned = []
        with open(fileName, 'r', encoding='utf-8') as csvfile:

            reader = csv.reader(csvfile, delimiter='~', quotechar='|')

            question_max_len = 0
            answer_max_len = 0
            i = 0

            max_len_question_index = 0
            max_len_answer_index = 0

            for row in reader:

                if (row[0] == 'eshop'):
                    continue

                question = self.processText(row[3])

                answer = self.processText(row[4])


                if(len(question) > self.maxQuestionLen):

                    eos = question[len(question)-1]
                    question = question[0:self.maxQuestionLen-1]
                    question.append(eos)

                if(len(answer)> self.maxAnswerLen):
                    eos = answer[len(answer)-1]

                    answer = answer[0:self.maxAnswerLen-1]
                    answer.append(eos)


                if (len(question) >= question_max_len):
                    question_max_len = len(question)
                    max_len_question_index = i

                if (len(answer) >= answer_max_len):
                    answer_max_len = len(answer)
                    max_len_answer_index = i

                i = i +1


                cleaned.append((question,answer))

        print('cleaned length', len(cleaned))
        print('question max length',question_max_len, max_len_question_index)
        #print(cleaned[max_len_question_index][0])
        print('answer max length',answer_max_len, max_len_answer_index)
        #print(cleaned[max_len_answer_index][0])

        outputFile = open("output.csv",'w', encoding='utf-8')
        writer = csv.writer(outputFile, delimiter='~')


        for row in cleaned:
            targetRow = []
            question = ' '.join(row[0])
            answer = ' '.join(row[1])
            targetRow.append(question)
            targetRow.append(answer)

        

            writer.writerow(targetRow)



        return cleaned

    def processText(self, text):

        text = text.replace('.',' ').replace(',',' ').replace('  ',' ')




        # cleans the text (numbers, links, etc)
        text = TextUtil.cleanText(text)

        # remove punctuation
        text = TextUtil.removePunctuaction(text);

        # removes stop words, which are not usefull to the mode
        text = TextUtil.stripStopWords(text)

        # remove names of the people
        text = TextUtil.removeNames(text)

        text.insert(0,'_start_')
        text.append('_end_')




        return text

