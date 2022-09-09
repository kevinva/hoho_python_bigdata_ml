#!/usr/bin/python3    # 1. 必须要加这句！

import sys
import nltk

word2count = {}

for line in sys.stdin:
    line = line.strip()
    words = filter(lambda word: word, line.split())
    for word in words:
        # print(f'{word}\t1')   # 2. 不能用print!!! 要用sys.stdout
        sys.stdout.write(f'{word}\t1\n')  

            