#!/usr/local/bin/python3.10

import sys

fruits ={}
for line in sys.stdin:
    line = line.strip()
    fruit, count = line.split(":", 1)
    count = int(count)
    fruits[fruit] = fruits.get(fruit, 0) + count

for fruit, count in fruits.items():
    print(f"{fruit}:{count}")
    # sys.stdout.write(f"{fruit}:{count}\n")