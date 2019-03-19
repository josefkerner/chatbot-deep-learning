import csv
import re
import numpy as np
from numpy import array
class TextUtil:

    def __init__(self):
        pass
    @staticmethod
    def iterative_levenshtein(s, t):
        """
            gets distance between source string and target string in term of necessary character swaps to be perfomed in order to get target string from the source string
            it is written in the iterative manner
        """
        rows = len(s) + 1
        cols = len(t) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for i in range(1, rows):
            dist[i][0] = i
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution

        return dist[row][col]

    # one hot encode method, more efficient in terms of memory than the Keras implementation (custom solution)

    @staticmethod
    def one_hot_encode(y, max_int):


        # resulting numpy array has shape:
        # (number of samples, max length of sample, length of vocabulary)
        # for example : (2000, 20, 26458)
        yenc = []
        for seq in y:

            pattern = []

            for index in seq:
                vector = np.zeros((max_int,), dtype=np.int8)
                vector[index - 1] = 1

                # pattern = np.vstack((pattern,vector))
                pattern.append(vector)

            yenc.append(pattern)

        yenc = array(yenc)

        print('bytes used for matrix one-hot encoding:', yenc.nbytes)
        return yenc

    # removes punctuation from the text and replaces it with english characters
    @staticmethod
    def removePunctuaction(text):

        punc = ["ř",'č',"ď","š","ě","á","í","é","ý","ž","ó",'ú','ů','ň']
        puncReplace = ['r','c','d','s','e','a','i','e','y','z','o','u','u','n']

        newText = []

        for word in text:
            word = str(word).lower()
            newWord = ''
            for char in word:
                if(char in punc):
                    newWord = newWord + puncReplace[punc.index(char)]
                else:
                    newWord = newWord + char

            newText.append(newWord)



        return newText

    # saves data into CSV file
    @staticmethod
    def saveCSV(data, path):
        if (len(data) == 0):
            return False

        header = list(data[0].keys())
        path = path

        with open(path, 'w', newline='',encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter='~')
            writer.writerow(header)

            for row in data:
                row = list(row.values())
                # print(row)

                writer.writerow(row)

    # removes most typical czech names from the text
    @staticmethod
    def removeNames(text):
        output = []
        names = ['martin','petr','miroslav','josef','nikola','libor','tomas','milan','zdenka','juri','daniel','vendula','honza','zdenek','pavlina','karel','lukas','martina'
            ,'dominik','katerina','jirka','zuzka','marek','filip','marcel','jiri','hana','stanislav','ladislav','anna','jaroslav','ivo','alena','pavel','eva',
                 'vasek','michal','adam','helca','vaclav','jana','ondra','roman','renata','lucie','lubos','vojta','sona','julie','david','dusan','adame',
                 'jakub','pepik','marian','patriku','michaela','anonyme','jaroslave','romana','jan','simona','martino','daniela','anonym','jarda','ales','lenka','majda','zdenko',
                 'veronika','petra','marketa','marketo','klara','klaro']
        for word in text:
            if('ova' in word or word in names):
                pass
            else:
                output.append(word)
        # use NameTag from CharlesUniversity
        return output

    @staticmethod
    # removes stop words which are not neccessary for the task at hand
    def stripStopWords(text):
            wordList = ['dobry', 'dekuju', 'prosim', 'nebo', 'kdyz', 'ale', 'o', 'den','cz','http','https','tel','html','pozdravem','team','pani','pane','vazeny','vazena','vazenya']
            parsedRow = []



            for word in text:
                if (word in wordList):
                    continue
                else:
                    if (word != ''):
                        parsedRow.append(word)

            return parsedRow

    # cleans text of non-alphabetical characters
    @staticmethod
    def cleanText(text):
        # strip numbers and other non alphabetic characters
        cleaned = []
        text = text.split(' ')
        for word in text:
            # matches only words in unicode string
            result = re.match('\w+', word, re.UNICODE)

            if(result):
                # get result group
                result = re.sub(r'[\d]','',result.group())
                if(result != ''):
                    cleaned.append(result)
        return cleaned