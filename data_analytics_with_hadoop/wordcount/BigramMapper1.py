#!/usr/bin/python3

import sys
# import nltk    # 注意第三方库在Hadoop集群上也要有
# import string

SEP = '\t'

def normalize(token):
    return token.lower()

def tokenize(value):
    # for token in nltk.wordpunct_tokenize(value):
    for token in value.split():
        token = normalize(token)
        # if not self.exclude(token):
        yield token

def main():
    for words in sys.stdin:
        # for bigram in nltk.bigrams(tokenize(words)):
        for bigram in tokenize(words):

            # print(bigram)
            # self.counter('words')
            # self.emit(bigram, 1)
            sys.stdout.write(f'{bigram}{SEP}{1}\n')


if __name__ == '__main__':
    main()