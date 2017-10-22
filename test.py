#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

with open("./data/ml-20m/genome-scores.csv", 'r') as f:
    f.readline()
    count = 1
    for line in f:
        line = line.rstrip('\n').split(',')
        if (count != int(line[1])):
            print ("%s miss information" % line[0])
        count += 1
        if (count == 1129):
            count = 1
pass