#!/opt/anaconda3/envs/hoho_RL/bin/python 

import sys
import nltk
import string

from framework import Mapper

class BigramMapper(Mapper):

    def __init__(self, infile=sys.stdin, separator='\t'):
        super(BigramMapper, self).__init__(infile, separator)

        self.stopwords = nltk.corpus.stopwords.words('english')
        self.punctuation = string.punctuation

    def exclude(self, token):
        return token in self.punctuation or token in self.stopwords

    def normalize(self, token):
        return token.lower()

    def tokenize(self, value):
        for token in nltk.wordpunct_tokenize(value):
            token = self.normalize(token)
            # if not self.exclude(token):
            yield token
    
    def map(self):
        for value in self:
            for bigram in nltk.bigrams(self.tokenize(value)):
                # print(bigram)
                self.counter('words')
                self.emit(bigram, 1)

if __name__ == '__main__':
    mapper = BigramMapper()
    mapper.map()