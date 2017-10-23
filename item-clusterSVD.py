#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from data_manager import Data_Factory
from sklearn.cluster import KMeans
import pandas as pd
from numpy.random import random

class SVD():
    def __init__(self, X, k=20):
        self.X = np.array(X)
        self.k = k
        self.ave = np.mean(self.X[:, 2])
        print ("the input data size is ", self.X.shape)
        self.bu = {}
        self.bi = {}
        self.pu = {}
        self.qi = {}
        self.movie_user = {}
        self.user_movie = {}
        for i in range(self.X.shape[0]):
            uid, mid, rat = self.X[i][0], self.X[i][1], self.X[i][2]
            self.movie_user.setdefault(mid, {})
            self.user_movie.setdefault(uid, {})
            self.movie_user[mid][uid] = rat
            self.user_movie[uid][mid] = rat
            self.bu.setdefault(uid, 0)
            self.bi.setdefault(mid, 0)
            self.pu.setdefault(uid, random((self.k, 1))/10*(np.sqrt(self.k)))
            self.qi.setdefault(mid, random((self.k, 1))/10*(np.sqrt(self.k)))

    def pred(self, uid, mid):
        self.bu.setdefault(uid, 0)
        self.bi.setdefault(mid, 0)
        self.pu.setdefault(uid, np.zeros((self.k, 1)))
        self.qi.setdefault(mid, np.zeros((self.k, 1)))
        score = self.ave + self.bu[uid] + self.bi[mid] + np.sum(self.pu[uid] * self.qi[mid])
        if score > 5:
            return 5
        if score < 1:
            return 1
        return score

    def train(self, steps=20, gamma=0.04, Lambda=0.15):
        for step in range(steps):
            print ("the ", step, "-th step is running")
            rmse_sum = 0.0
            kk = np.random.permutation(self.X.shape[0])
            for j in range(self.X.shape[0]):
                i = kk[j]
                uid = self.X[i][0]
                mid = self.X[i][1]
                rat = self.X[i][2]
                eui = rat - self.pred(uid, mid)
                rmse_sum += eui ** 2
                self.bu[uid] = self.bu[uid] + gamma * (eui - Lambda * self.bu[uid])
                self.bi[mid] = self.bi[mid] + gamma * (eui - Lambda * self.bi[mid])
                temp = self.pu[uid]
                self.pu[uid] = self.pu[uid] + gamma * (eui * self.qi[mid] - Lambda * self.pu[uid])
                self.qi[mid] = self.qi[mid] + gamma * (eui * temp - Lambda * self.qi[mid])
            gamma = gamma * 0.93
            print ("the rmse of this step on train data is ", np.sqrt(rmse_sum / self.X.shape[0]))

def pred_test(test_X, models, labels):
    sums = 0
    test_X = np.array(test_X)
    count = 0
    for i in range(test_X.shape[0]):
        if test_X[i][1] not in labels:
            continue
        count += 1
        pre = models[labels[test_X[i][1]]].pred(test_X[i][0], test_X[i][1])
        sums += (pre - test_X[i][2]) ** 2
    rmse = np.sqrt(sums / count)
    print ("the rmse on test data is ", rmse)
    return

def item_cluster(X, cluster_num):
    df_genome = X
    genome_shape = df_genome.shape
    kmeans = KMeans(n_clusters=cluster_num).fit(df_genome)
    labels = pd.Series(kmeans.labels_, index=df_genome.index)
    return labels




if __name__ == '__main__':
    a = Data_Factory()
    df_train = a.generate_genome()
    #df_train = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 2], [1, 3, 3], [2, 3, 4], [1, 2, 3]]))
    """
    R = a.read_rating('./data/ml-1m/ratings.dat')
    train, valid, test = a.generate_train_valid_test_file(R, 0.002)
    a.save(train, './data/ml-1m/0.002/train.dat')
    a.save(test, './data/ml-1m/0.002/test.dat')
    """
    train = a.load('./data/ml-1m/0.002/train.dat')
    test = a.load('./data/ml-1m/0.002/test.dat')
    cluster_num = 5
    labels = item_cluster(df_train, cluster_num)
    split_train = {}
    for i in range(cluster_num):
        split_train[i] = []
    for i in range(len(train)):
        if train[i][1] in labels:
            split_train[labels[train[i][1]]].append(train[i])
    models = {}
    for i in range(cluster_num):
        print ("the", i, "-th model is being trained")
        models[i] = SVD(split_train[i])
        models[i].train()
    pred_test(test, models, labels)
    pass