#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file CF.py
@author libo
@date 2017.12.4

This model calculates missing Ratings using Item-based Collaborative Filtering, and test it on RMSE and a Top-N Recommendation
"""

import numpy as np
from data_manager import Data_Factory
import pandas as pd
import time
from multiprocessing import Pool
import os

TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

class Item_Based_CF():
    def __init__(self, data):
        data = np.array(data)
        self.user_movie = {}
        self.movie_user = {}
        self.ave = np.mean(data[:, 2])
        for i in range(len(data)):
            uid, mid, rat = data[i][0], data[i][1], data[i][2]
            self.user_movie.setdefault(uid, {})
            self.movie_user.setdefault(mid, {})
            self.user_movie[uid][mid] = rat
            self.movie_user[mid][uid] = rat
        self.similarity = {}
        self.user_list = list(self.user_movie.keys())
        self.movie_list = list(self.movie_user.keys())
        pass

    def sim_cal(self, m1, m2):
        if m1 in self.movie_user and m2 in self.movie_user and m1 != m2:
            return self.sim_cf[m1][m2]
        else:
            return 0

    def pred(self, user, movie):
        item_list = self.user_movie.get(user, {})
        sim_acc = []
        rat_acc = []
        for item in item_list:
            sim = self.sim_cal(item, movie)
            if sim <= 0:
                continue
            rat_acc.append(self.user_movie[user][item])
            sim_acc.append(sim)
        rat_acc = np.array(rat_acc)
        sim_acc = np.array(sim_acc)
        rat = np.dot(rat_acc, sim_acc)
        sim = np.sum(sim_acc)

        if sim == 0:
            return self.ave
        return rat / sim

    def load_sim(self):
        self.sim_cf = pd.read_csv('./data/ml-1m/subprocess/sim_cf_merge')
        self.sim_cf = pd.Series(self.sim_cf['sim'].values,
                                pd.MultiIndex.from_arrays([self.sim_cf['mid1'].values, self.sim_cf['mid2'].values]))
        self.sim_cf = self.sim_cf.unstack(level=-1)
        return

    def test(self, test_rat):
        test_rat = np.array(test_rat)
        n = test_rat.shape[0]
        err_square = 0.0
        output = []
        for i in range(n):
            pred_rat = self.pred(test_rat[i][0], test_rat[i][1])
            err_square += (pred_rat - test_rat[i][2]) ** 2
            res = list(test_rat[i])
            res.append(pred_rat)
            output.append(res)
            if i % 100 == 0:
                print ("processing items quantity: %d" % i)
        print ("the rmse on test data is: %f" % (err_square / n))
        return output

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
        file = './data/ml-1m/subprocess/rat_res_cf_' + str(os.getpid())
        with open(file, 'w') as f:
            for i in range(len(rat_res)):
                f.write("%s,%s,%s\n" % (str(rat_res[i][0]), str(rat_res[i][1]), str(rat_res[i][2])))
        print("[%s]subprocess %s done." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
        return

    def test_from_all_ratings(self, test_rat):
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_itemcf', header=None, names=['uid', 'mid', 'rat'])
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
                    #pred.append(rat_res.loc[uid][mid])
                    pred.append(self.ave)
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

    def top_n(self, test, n=10):
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_itemcf', header=None, names=['uid', 'mid', 'rat'])
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

        k_num = 0
        test_num = 0
        common_num = 0

        count = 0
        for uid in test_movie:
            if uid in rat_res.index:
                threshold = rat_res.loc[uid].sample(500).sort_values(ascending=False).iloc[n-1]
                for mid in test_movie[uid]:
                    if uid in rat_res.index and mid in rat_res.columns:
                        test_num += 1
                        if rat_res.loc[uid][mid] >= threshold:
                            common_num += 1
                k_num += n
                count += 1
                if count % 100:
                    print("process quantity:", count)

        precise = common_num / k_num
        recall = common_num / test_num

        print("precise is ", precise)
        print("recall is ", recall)
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
    b = Item_Based_CF(train)
    b.load_sim()
    b.gen_all_ratings(32)
    b.test_from_all_ratings(test)
    b.top_n(test, 40)
    pass