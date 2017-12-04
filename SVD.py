#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data_manager import Data_Factory
from numpy.random import random
from multiprocessing import Pool
import time
import pandas as pd
import os

TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

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
        self.user_list = list(self.user_movie.keys())
        self.movie_list = list(self.movie_user.keys())

    def pred(self, uid, mid):
        self.bu.setdefault(uid, 0)
        self.bi.setdefault(mid, 0)
        self.pu.setdefault(uid, np.zeros((self.k, 1)))
        self.qi.setdefault(mid, np.zeros((self.k, 1)))
        score = self.ave + self.bu[uid] + self.bi[mid] + np.sum(self.pu[uid] * self.qi[mid])
        """
        if score > 5:
            return 5
        if score < 1:
            return 1
        """
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

    def test(self, test_X):
        output = [] 
        sums = 0
        test_X = np.array(test_X)
        for i in range(test_X.shape[0]):
            pre = self.pred(test_X[i][0], test_X[i][1])
            output.append(pre)
            sums += (pre - test_X[i][2]) ** 2
        rmse = np.sqrt(sums / test_X.shape[0])
        print ("the rmse on test data is ", rmse)
        return output

    def childProcess(self, begin, end):
        print("[%s]subprocess %s begin." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
        rat_res = []
        for i in range(begin, end):
            uid = self.user_list[i]
            for mid in self.movie_list:
                if mid in self.user_movie[uid]:
                    continue
                pred = self.pred(uid, mid)
                rat_res.append([uid, mid, pred])
        file = './data/ml-1m/subprocess/rat_res_svd_' + str(os.getpid())
        with open(file, 'w') as f:
            for i in range(len(rat_res)):
                f.write("%s,%s,%s\n" % (str(rat_res[i][0]), str(rat_res[i][1]), str(rat_res[i][2])))
        print("[%s]subprocess %s done." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
        return

    def gen_all_ratings(self, k=32):
        n = len(self.user_list)
        index = [int(n * i / k) for i in range(k)]
        index.append(n)
        p = Pool(k)
        for i in range(k):
            p.apply_async(self.childProcess, args=(index[i], index[i + 1]))
        print("[%s]Waiting for all subprocesses done..." % time.strftime(TIMEFORMAT, time.localtime()))
        p.close()
        p.join()
        print("[%s]All subprocesses done." % time.strftime(TIMEFORMAT, time.localtime()))
        return

    def test_from_all_ratings(self, test_rat):
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_svd', header=None, names=['uid', 'mid', 'rat'])
        rat_res = pd.Series(rat_res['rat'].values,
                            pd.MultiIndex.from_arrays([rat_res['uid'].values, rat_res['mid'].values]))
        rat_res = rat_res.unstack(level=-1)
        test_rat = np.array(test_rat)
        n = test_rat.shape[0]

        pred = []
        rat_val = []
        count = 0
        for i in range(n):
            uid = test_rat[i][0]
            mid = test_rat[i][1]
            rat = test_rat[i][2]
            if uid in rat_res.index and mid in rat_res.columns:
                try:
                    pred.append(rat_res.loc[uid][mid])
                    rat_val.append(rat)
                    count += 1
                    if count % 1000 == 0:
                        print("process quantity: ", count)
                except:
                    print("key error", uid, mid)

        pred = np.array(pred)
        rat_val = np.array(rat_val)
        rmse = np.sum((pred - rat_val) ** 2)

        precise = np.sqrt(rmse / count)
        print("the rmse on test data is: ", precise)
        print("the number is: ", count)
        return precise

    def top_n(self, test):
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_svd', header=None, names=['uid', 'mid', 'rat'])
        rat_res = pd.Series(rat_res['rat'].values,
                            pd.MultiIndex.from_arrays([rat_res['uid'].values, rat_res['mid'].values]))
        rat_res = rat_res.unstack(level=-1)

        test = np.array(test)
        num = test.shape[0]
        test_movie = {}
        for i in range(num):
            uid = test[i][0]
            mid = test[i][1]
            rat = test[i][2]
            if rat == 5.0:
                test_movie.setdefault(uid, [])
                test_movie[uid].append(mid)

        n = [1]
        for i in range(20):
            n.append(5 * (i + 1))
        k_num = [0 for i in n]
        test_num = [0 for i in n]
        common_num = [0 for i in n]

        count = 0
        for uid in test_movie:
            if uid in rat_res.index:
                threshold = rat_res.loc[uid].sample(500).sort_values(ascending=False)
                for mid in test_movie[uid]:
                    if uid in rat_res.index and mid in rat_res.columns:
                        for i in range(len(n)):
                            test_num[i] += 1
                            if rat_res.loc[uid][mid] >= threshold.iloc[n[i] - 1]:
                                common_num[i] += 1
                for i in range(len(n)):
                    k_num[i] += n[i]
                count += 1
                if count % 100:
                    print("process quantity:", count)

        precise = np.array(common_num) / np.array(k_num)
        recall = np.array(common_num) / np.array(test_num)

        print("precise is ", list(precise))
        print("recall is ", list(recall))
        return

if __name__ == '__main__':
    a = Data_Factory()
    """
    R = a.read_rating('./data/ml-1m/ratings.dat')
    train, valid, test = a.generate_train_valid_test_file(R, 0.002)
    a.save(train, './data/ml-1m/0.002/train.dat')
    a.save(test, './data/ml-1m/0.002/test.dat')
    """
    train = a.load('./data/ml-1m/0.25/train.dat')
    test = a.load('./data/ml-1m/0.25/test.dat')
    b = SVD(train)
    b.train()
    b.gen_all_ratings()
    b.test_from_all_ratings(test)
    b.top_n(test)
    pass
