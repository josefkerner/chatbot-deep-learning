import glob
import csv

from lib.textUtil import TextUtil

class dataCombiner:
    def __init__(self):
        files = glob.glob('output/*.csv')
        self.counter = 0
        self.evaluationCounter = 0
        self.data = []
        self.finalData = []
        for file in files:

            self.readFile(file)

        self.processData()

    def processData(self):
        withoutCount = 0
        rowCounter = 0
        withCount = 0
        for row in self.data:
            if(row['response'] == 'NULL'):
                withoutCount += 1
            self.evaluatePost(rowCounter)

            actualRow = self.data[rowCounter]
            if (actualRow['response'] != 'NULL' and actualRow['response'] != 'response'):

                dist = TextUtil.iterative_levenshtein(actualRow['question'],actualRow['response'])

                if(dist > 3):
                    self.finalData.append(self.data[rowCounter])
                withCount +=1
            rowCounter +=1
        print(self.finalData)
        TextUtil.saveCSV(self.finalData,'heureka_train.csv')


        print('all posts',rowCounter)

        print('post written by customer care',self.evaluationCounter)
        print(withoutCount, withCount)
        print(len(self.finalData))
    def readFile(self, fileName):
        #print(fileName)
        with open(fileName, 'r', encoding='utf-8') as csvfile:

            reader = csv.reader(csvfile, delimiter='~', quotechar='|')
            eshopName = fileName.split('_')[1].replace('.csv', '')
            for row in reader:
                sample = {'eshop':eshopName,'date': row[0], 'author': row[1], 'question': row[2], 'response': row[3]}
                self.data.append(sample)

    def evaluatePost(self, index):
        sample = self.data[index]
        if(len(self.data) > index+1):
            prev = self.data[index+1]
        else:
            return True

        distance = TextUtil.iterative_levenshtein(sample['eshop'].lower(), sample['author'].lower())

        if(distance <3 or 's.r.o' in sample['author'] or 'a.s.' in sample['author']):
            # post was written by customer care
            self.evaluationCounter += 1
            print(sample['author'],'-----------------------------------------------------------')
            if(prev['eshop'] == sample['eshop']):
                print(sample['question'])
                print(prev['question'])
                self.data[index+1]['response'] = sample['question']



dataCombiner()