#!/opt/anaconda3/envs/hoho_RL/bin/python    # 1. 必须要加这句！ 这句声明表明要用哪个python解释器。不加的话执行该文件时要用语法'python xxx.py'，若加上则执行该文件只要'./xxx.py'


from argparse import _CountAction
from operator import itemgetter
import sys

word2count = {}

for line in sys.stdin:
    line = line.strip()
    # print(line)
    word, count = line.split('\t', 1)
    try:
        count = int(count)
        word2count[word] = word2count.get(word, 0) + count
    except ValueError:
        pass

sorted_word2count = sorted(word2count.items(), key=itemgetter(0))

for word , count in sorted_word2count:
    # print(f'{word}\t{count}')    # 2. 不能用print!!! 要用sys.stdout
    sys.stdout.write(f'{word}\t{count}\n') 