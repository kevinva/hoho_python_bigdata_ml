#!/usr/bin/python3

import sys
from itertools import groupby
from operator import itemgetter

SEP = '\t'

def read_mapper_output(file):
    for line in file:
        yield line.rstrip().split(SEP, 1)

def main():
    data = read_mapper_output(sys.stdin)
    for bigram, group in groupby(data, itemgetter(0)):
        try:
           total_count = sum(int(count) for key, count in group)
           sys.stdout.write(f'{bigram}{SEP}{total_count}\n')
        except ValueError:
            pass

if __name__ == '__main__':
    main()