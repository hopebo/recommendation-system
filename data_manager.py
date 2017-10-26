#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_manager module: process input data
"""

import os
import sys
import pickle
import numpy as np
import random
import pandas as pd
import csv

class Data_Factory():
    def read_rating(self, path):
        user_movie_ratings = []
        with open(path, 'r') as f:
            for line in f:
                record = line.split("::")
                user_movie_ratings.append((int(record[0]), int(record[1]), float(record[2])))
        return user_movie_ratings

    def generate_train_valid_test_file(self, R, ratio):
        n = np.array(R).shape[0]
        valid_n = test_n = int(n * ratio / 2)
        random.shuffle(R)
        valid_ratings = R[0 : valid_n]
        test_ratings = R[valid_n : valid_n + test_n]
        train_ratings = R[valid_n + test_n : ]
        return train_ratings, valid_ratings, test_ratings

    def save(self, data, path):
        pickle.dump(data, open(path, 'wb'))
        return

    def load(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def generate_genome(self):
        df_genome = pd.read_csv('./data/ml-20m/genome-scores.csv', ',')
        df_genome = df_genome.set_index(['movieId', 'tagId']).unstack(level=-1)
        mid = []
        ori_mid = []
        exclude = []
        mid_map = {}

        with open('./data/ml-1m/mid_map', 'r') as f:
            for line in f:
                line = line.rstrip().split('\t')
                try:
                    corr_mid = int(line[-1])
                    mid_map[int(line[0])] = corr_mid
                except:
                    exclude.append(int(line[0]))

        with open('./data/ml-1m/movies.dat', 'rb') as f:
            for line in f:
                line = line.decode('latin1').split("::")
                cur_mid = int(line[0])
                if cur_mid not in exclude:
                    ori_mid.append(cur_mid)
                    if cur_mid in mid_map:
                        mid.append(mid_map[cur_mid])
                    else:
                        mid.append(cur_mid)

        df_train = df_genome.loc[mid]
        df_train['ori_mid'] = ori_mid
        df_train.dropna(axis=0, how='any', inplace=True)
        df_train.index = df_train['ori_mid']
        df_train = df_train.drop('ori_mid', axis=1)
        return df_train

    def generate_train_valid_test_file_with_remove(self, data, ratio, df_train):
        R = []
        for i in range(len(data)):
            if data[i][1] in df_train.index:
                R.append(data[i])
        n = np.array(R).shape[0]
        valid_n = test_n = int(n * ratio / 2)
        random.shuffle(R)
        valid_ratings = R[0 : valid_n]
        test_ratings = R[valid_n : valid_n + test_n]
        train_ratings = R[valid_n + test_n : ]
        return train_ratings, valid_ratings, test_ratings


if __name__ == '__main__':
    a = Data_Factory()
    R = a.read_rating('./data/ml-1m/ratings.dat')
    user_dict = {}
    movie_dict = {}
    for i in range(len(R)):
        user_dict[R[i][0]] = 1
        movie_dict[R[i][1]] = 1
    print (len(user_dict))
    print (len(movie_dict))
    train, valid, test = a.generate_train_valid_test_file(R, 0.002)
    pass