#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file data_manager.py
@author libo
@date 2017.12.4

This module processes input data, splits it into train set and test set, generates Tag Genome DataFrame(removing the not included
movie items.
"""

import pickle
import numpy as np
import random
import pandas as pd

class Data_Factory():
    def read_rating(self, path):
        """
        @brief: process input data
        """
        user_movie_ratings = []
        with open(path, 'r') as f:
            for line in f:
                record = line.split("::")
                user_movie_ratings.append((int(record[0]), int(record[1]), float(record[2])))
        return user_movie_ratings

    def generate_train_test_file(self, R, ratio):
        """
        @brief: split input data into train set and test set
        """
        n = np.array(R).shape[0]
        test_n = int(n * ratio)
        random.shuffle(R)
        test_ratings = R[0 : test_n]
        train_ratings = R[test_n : ]
        print("train size is %d" % len(train_ratings))
        user = {}
        movie = {}
        for i in range(len(train_ratings)):
            user.setdefault(train_ratings[i][0], 0)
            movie.setdefault(train_ratings[i][1], 0)
        print("user number in train set is %d" % len(user))
        print("movie number in train set is %d" % len(movie))
        print("test size is %d" % len(test_ratings))
        user = {}
        movie = {}
        for i in range(len(test_ratings)):
            user.setdefault(test_ratings[i][0], 0)
            movie.setdefault(test_ratings[i][1], 0)
        print("user number in test set is %d" % len(user))
        print("movie number in test set is %d" % len(movie))
        return train_ratings, test_ratings

    def save(self, data, path):
        """
        @brief: save data set
        """
        pickle.dump(data, open(path, 'wb'))
        return

    def load(self, path):
        """
        @brief: load data set
        """
        data = pickle.load(open(path, 'rb'))
        return data

    def generate_genome(self):
        """
        @brief: generate Tag Genome DataFrame
        """
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
        """
        @biref: remove data items whose movie is not included in Tag Genome Data
        """
        R = []
        for i in range(len(data)):
            if data[i][1] in df_train.index:
                R.append(data[i])
        n = np.array(R).shape[0]
        test_n = int(n * ratio)
        random.shuffle(R)
        test_ratings = R[0:test_n]
        train_ratings = R[test_n:]
        return train_ratings, test_ratings


if __name__ == '__main__':
    a = Data_Factory()
    R = a.read_rating('./data/ml-1m/ratings.dat')
    print("original ratings' size is %d" % len(R))
    df_train = a.generate_genome()
    R_remove = []
    user_dict = {}
    movie_dict = {}
    for i in range(len(R)):
        if R[i][1] in df_train.index:
            user_dict[R[i][0]] = 1
            movie_dict[R[i][1]] = 1
            R_remove.append(R[i])
    print("users' size is %d" % len(user_dict))
    print("items' size is %d" % len(movie_dict))
    train, test = a.generate_train_test_file(R_remove, 0.25)
    a.save(train, './data/ml-1m/0.25/train.dat')
    a.save(test, './data/ml-1m/0.25/test.dat')
    pass