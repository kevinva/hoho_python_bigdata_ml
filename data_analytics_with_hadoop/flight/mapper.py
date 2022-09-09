#!/opt/anaconda3/envs/hoho_RL/bin/python

import sys
import csv

SEP = '\t'

class Mapper(object):

    def __init__(self, stream, sep=SEP):
        self.stream = stream
        self.sep = sep

    def emit(self, key, value):
        sys.stdout.write(f'{key}{self.sep}{value}')

    def map(self):
        for row in self:
            parts = row[0].split('\t')
            print(parts[6])
            self.emit(parts[3], parts[6])
        
    def __iter__(self):
        reader = csv.reader(self.stream)
        for row in reader:
            yield row

if __name__ == '__main__':
    mapper = Mapper(sys.stdin)
    mapper.map()