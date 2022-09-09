#!/usr/bin/python3 

from framework import Reducer

class SumReducer(Reducer):

    def reduce(self):
        bigram_count = {}
        for key, values in self:
            if key in bigram_count:
                for v in values:
                    bigram_count[key] = bigram_count.get(key, 0) + int(v[1])
            else:
                bigram_count[key] = 1
            
        for key, count in bigram_count.items():
            self.emit(key, count)

if __name__ == '__main__':
    reducer = SumReducer()
    reducer.reduce()


