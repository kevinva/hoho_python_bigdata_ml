#!/usr/local/bin/python3.10

import sys

for line in sys.stdin:
    line = line.strip()
    fruit, count = line.split(',', 1)
    print(f"{fruit}:{count}")