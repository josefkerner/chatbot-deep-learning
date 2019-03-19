# import libraries
from urllib.request import urlopen
import urllib.request, urllib.error
from lib.textUtil import TextUtil
from bs4 import BeautifulSoup
import csv
import os

class dataScraper:
    def __init__(self, sourceName):

        if(sourceName == 'heureka'):
            self.data = []

            self.eshop_start_num = 0
            self.eshop_max_num = 400
            self.getEshops()

    # saves parsed data for one eshop into a csv file
    def saveDiscussion(self,eshop_page_num):

        if(len(self.data) == 0):
            return False

        header = list(self.data[0].keys())


        with open('output/'+str(eshop_page_num)+'_'+self.eshop+'.csv', 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter='~')
            writer.writerow(header)

            for row in self.data:
                row = list(row.values())
                #print(row)

                writer.writerow(row)

    # gets all the eshops on heureka and starts downloading data from them
    def getEshops(self):

        for eshop_page_num in range(self.eshop_start_num,self.eshop_max_num):
            url = 'https://obchody.heureka.cz/elektronika/?f='+str(eshop_page_num)
            soup = self.getSoup(url)

            if(soup == False):
                print('eshop pages finished')
                break

            table = soup.find('tbody')
            trs = table.find_all('tr')

            for tr in trs:
                a = tr.find('a', attrs={'class': 'e-button--simple'}, href=True)

                self.eshop = str(a['href']).split('/')[1]

                print('Downloading data for eshop',self.eshop)
                self.downloadSourceHeureka()
                self.saveDiscussion(eshop_page_num)
                self.data = []

    # downloads a selected ecommerce site from heureka
    def downloadSourceHeureka(self):
        postsProcessed = 0
        for x in range(1, self.eshop_max_num):
            print("processing page ",x)

            soup = self.getSoup('https://obchody.heureka.cz/'+self.eshop+'/diskuse/?f='+str(x)+'#filtr')

            if(soup == False):
                print('page does not exist')
                break
            postsCount = self.parseHeurekaPage(soup)

            postsProcessed = postsProcessed + postsCount

        print(postsProcessed)

    # parses a single heureka page and reads posts from it

    def parseHeurekaPage(self, soup):
        ul = soup.find('ul', attrs={'class': 'c-box-list'})
        postsCount = 0


        for li in ul.find_all('li', recursive=False):
            p = li.find('p', attrs={'class': 'c-post__summary'})

            row = {}
            time = li.find('time').find(text=True, recursive=False).strip()

            row['time'] = time

            author = self.getAuthor(li)
            row['author'] = author

            question = p.get_text().replace('\n','').replace('            ','').strip()

            if('Příspěvek byl provozovatelem odstraněn' not in question and question !=''):
                row['question'] = question
            else:
                continue

            response = li.find('div', attrs={'class' : 'c-post__response'})
            if(response != None):


                row['response'] = self.parseEmbeddedResponse(response)


            else:
                row['response'] = 'NULL'

            postsCount = postsCount + 1

            self.data.append(row)

        return postsCount

    # get author of the post

    def getAuthor(self,element):
        author = element.find('p', attrs={'class': 'c-post__author'}).find(text=True, recursive=False).strip()
        return author

    # parses response if its embedded within the question

    def parseEmbeddedResponse(self, response):
        # can be multiple answers per single question
        responseLis = response.find_all('li', attrs={'class': 'c-post'})

        response = 'NULL'
        for li in responseLis:
            # find out if eshop is author of the response

            author = self.getAuthor(li)


            distance = TextUtil.iterative_levenshtein(str(self.eshop).lower(), author.lower())

            if(distance <= 4 or 's.r.o' in author or 'a.s.' in author): # if author name differs maximum of three characters from the eshop name
                responseText = li.find('p', attrs={'class' : 'c-post__summary'}).get_text().strip()
                if(response == 'NULL'):
                    response = responseText
                else:
                    response = response+' '+responseText
        print(response)

        return response

    # gets a beautiful soup object

    def getSoup(self, url):

        try:
            page = urlopen(url)
        except urllib.error.HTTPError:
            return False
        soup = BeautifulSoup(page, 'html.parser')

        return soup


dataScraper('heureka')