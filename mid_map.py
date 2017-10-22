#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
find mid map relationship between ml-1m movies.dat and ml-20m movies.csv
"""

import matplotlib.pyplot as plt
import numpy as np

mid_20m = {}
with open('./data/ml-20m/movies.csv', 'r') as f:
    f.readline()
    for line in f:
        first = line.find(',')
        second = line.rfind(',')
        mid = int(line[:first])
        movie = line[first+1:second].strip('"')
        mid_20m[mid] = movie

mid_1m = {}
with open('./data/ml-1m/movies.dat', 'rb') as f:
    for line in f:
        line = line.decode('latin1').split("::")
        mid_1m[int(line[0])] = line[1]

for mid in mid_1m:
    if (mid not in mid_20m) or (mid_1m[mid] != mid_20m[mid]):
        movie = mid_20m.get(mid, "doesn't exist!")
        words = mid_1m[mid].split(' ')
        bag = movie.split(' ')
        count = 0
        for word in words:
            if word in bag:
                count += 1
        if count / len(words) < 0.5:
            print("%d\t%s\t%s" % (mid, mid_1m[mid], movie))
pass