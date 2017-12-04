#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file ICCF.py(Item Clustering Collaborative Filtering)
@author libo
@date 2017.12.4

This module provides an ICCF algorithm implement, including previous preparing functions, train(), calculating all the ratings
using trained model, test the model on RMSE and a Top-N Recommendation
"""

import numpy as np
from data_manager import Data_Factory
from sklearn.cluster import KMeans
import pandas as pd
import time
from numpy.random import random
from multiprocessing import Pool
import os

TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

class Item_Based_CF():
    def __init__(self, data):
        self.data = np.array(data)
        self.user_movie = {}
        self.movie_user = {}
        self.ave = np.mean(self.data[:, 2])
        self.bu = {}
        self.bi = {}
        self.gu = {}
        for i in range(len(self.data)):
            uid, mid, rat = int(self.data[i][0]), int(self.data[i][1]), float(self.data[i][2])
            self.user_movie.setdefault(uid, {})
            self.movie_user.setdefault(mid, {})
            self.bu.setdefault(uid, 0)
            self.bi.setdefault(mid, 0)
            self.gu.setdefault(uid, random())
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
        """
        @brief: predict a user's preference for a movie
        """
        self.bu.setdefault(user, 0)
        self.bi.setdefault(movie, 0)
        item_list = self.user_movie.get(user, {})
        sim_acc = 0.0
        rat_acc = 0.0
        sim_acc_cluster = 0.0
        rat_acc_cluster = 0.0

        sim_cluster = []
        rat_cluster = []
        bi_cluster = []

        sim_cf = []
        rat_cf = []
        bi_cf = []

        for item in item_list:
            if movie in self.index and item in self.clusters[self.labels[movie]]:
                sim = self.sim_cal_cluster(item, movie)
                sim_cluster.append(sim)
                rat_cluster.append(self.user_movie[user][item])
                bi_cluster.append(self.bi[item])

            else:
                sim = self.sim_cal(item, movie)
                if sim <= 0:
                    continue
                sim_cf.append(sim)
                rat_cf.append(self.user_movie[user][item])
                bi_cf.append(self.bi[item])

        sim_cluster = np.array(sim_cluster)
        rat_cluster = np.array(rat_cluster)
        bi_cluster = np.array(bi_cluster)
        sim_cf = np.array(sim_cf)
        rat_cf = np.array(rat_cf)
        bi_cf = np.array(bi_cf)

        rat_acc_cluster = np.dot(sim_cluster, rat_cluster - bi_cluster - self.ave - self.bu[user])
        sim_acc_cluster = np.sum(sim_cluster)
        rat_acc = np.dot(sim_cf, rat_cf - bi_cf - self.ave - self.bu[user])
        sim_acc = np.sum(sim_cf)
        pred_cf = 0.0
        pred_cluster = 0.0
        if sim_acc != 0:
            pred_cf = rat_acc / sim_acc
        if sim_acc_cluster != 0:
            pred_cluster = rat_acc_cluster / sim_acc_cluster
        print("ave: %f, bu: %f, bi: %f, gu: %f, cf: %f, cluster: %f" % (self.ave, self.bu[user], self.bi[movie], self.gu[user], pred_cf, pred_cluster))
        return self.ave + self.bu[user] + self.bi[movie] + self.gu[user]*pred_cf + (1-self.gu[user])*pred_cluster, pred_cf, pred_cluster

    def train(self, steps=20, gamma=0.04, Lambda=0.15):
        """
        @brief: use stochastic gradient descent to train model, using a tip which goes through data items parameters
        involved haven't been updated in current loop.
        :param steps: loop number of stochastic gradient descent
        :param gamma: update step length, shrunk after each loop
        :param Lambda: L2 regularization
        """
        for step in range(steps):
            print ("the ", step, "-th step is running")
            rmse_sum = 0.0
            user_dict = {}
            movie_dict = {}
            kk = np.random.permutation(self.data.shape[0])
            count = 0
            for j in range(self.data.shape[0]):
                i = kk[j]
                if j % 10000 == 0:
                    print("[%s]GD %d -th train data" % (time.strftime(TIMEFORMAT, time.localtime()), j))
                uid = self.data[i][0]
                mid = self.data[i][1]
                rat = self.data[i][2]
                if uid in user_dict and mid in movie_dict:
                    continue
                count += 1
                user_dict.setdefault(uid, 0)
                movie_dict.setdefault(mid, 0)
                pred, pred_cf, pred_cluster = self.pred(uid, mid)
                eui = pred - rat
                rmse_sum += eui ** 2
                self.bu[uid] = self.bu[uid] + gamma * (-eui - Lambda * self.bu[uid])
                self.bi[mid] = self.bi[mid] + gamma * (-eui - Lambda * self.bi[mid])
                self.gu[uid] = self.gu[uid] + gamma * (-eui * (pred_cf - pred_cluster) - Lambda * self.gu[uid])
            gamma = gamma * 0.93
            print("%d records have been processed" % count)
            print("the rmse of this step on train data is ", np.sqrt(rmse_sum / count))

    def item_cluster(self, X, cluster_num):
        """
        @brief: use K-means algorithm to cluster movies by Tag Genomes
        :param X: Tag Genomes(DataFrame)
        :param cluster_num: cluster number k
        """
        self.df_genome = X
        self.genome_shape = self.df_genome.shape
        kmeans = KMeans(n_clusters=cluster_num).fit(self.df_genome)
        self.labels = pd.Series(kmeans.labels_, index=self.df_genome.index)
        self.index = set(self.labels.index)
        self.clusters = {}
        for label in range(cluster_num):
            self.clusters[label] = []
        for mid, label in self.labels.iteritems():
            self.clusters[label].append(mid)
        for label in self.labels:
            self.clusters[label] = set(self.clusters[label])
        return

    def sim_cal_cluster(self, m1, m2):
        sim = 0.0
        if m1 > m2:
            sim = self.sim_cluster.loc[m2][m1]
        elif m1 < m2:
            sim = self.sim_cluster.loc[m1][m2]
        return sim

    def load_sim(self, df_train):
        """
        @brief: load similarities by Ratings(self.sim_cf) and Tag Genomes(self.sim_cluster) which have been calculated
        previously offline
        """
        self.sim_cf = pd.read_csv('./data/ml-1m/subprocess/sim_cf_merge')
        self.sim_cf = pd.Series(self.sim_cf['sim'].values,
                                pd.MultiIndex.from_arrays([self.sim_cf['mid1'].values, self.sim_cf['mid2'].values]))
        self.sim_cf = self.sim_cf.unstack(level=-1)

        self.sim_cluster = pd.read_csv('./data/ml-1m/subprocess/sim_cluster_merge')
        self.sim_cluster = pd.Series(self.sim_cluster['sim'].values,
                                pd.MultiIndex.from_arrays([self.sim_cluster['mid1'].values, self.sim_cluster['mid2'].values]))
        self.sim_cluster = self.sim_cluster.unstack(level=-1)
        return

    def childProcess(self, begin, end):
        print("[%s]subprocess %s begin." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
        rat_res = []
        for i in range(begin, end):
            uid = self.user_list[i]
            for mid in self.movie_list:
                if mid in self.user_movie[uid]:
                    continue
                pred, pred_cf, pred_cluster = self.pred(uid, mid)
                rat_res.append([uid, mid, pred])
        file = './data/ml-1m/subprocess/rat_res_' + str(os.getpid())
        with open(file, 'w') as f:
            for i in range(len(rat_res)):
                f.write("%s,%s,%s\n" % (str(rat_res[i][0]), str(rat_res[i][1]), str(rat_res[i][2])))
        print("[%s]subprocess %s done." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
        return

    def test_from_all_ratings(self, test_rat):
        """
        @brief: calculate RMSE on test set
        """
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_0.25', header=None, names=['uid', 'mid', 'rat'])
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
                    pred_rat = rat_res.loc[uid][mid]
                    if pred_rat > 5:
                        pred_rat = 5
                    elif pred_rat < 1:
                        pred_rat = 1
                    pred.append(pred_rat)
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

    def gen_all_ratings(self, k=32):
        """
        @brief: generate all missing ratings by above trained model using multiprocessing
        """
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

    def top_n(self, test):
        """
        @brief: calculate Recall and Precision Rate in a top-n Recommendation
        """
        rat_res = pd.read_csv('./data/ml-1m/0.25/merge_rat_res_0.25', header=None, names=['uid', 'mid', 'rat'])
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
            n.append(5*(i+1))
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
                            if rat_res.loc[uid][mid] >= threshold.iloc[n[i]-1]:
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
    df_train = a.generate_genome()
    """
    R = a.read_rating('./data/ml-1m/ratings.dat')
    train, valid, test = a.generate_train_valid_test_file(R, 0.25)
    a.save(train, './data/ml-1m/0.25/train.dat')
    a.save(test, './data/ml-1m/0.25/test.dat')
    """
    train = a.load('./data/ml-1m/0.25/train.dat')
    test = a.load('./data/ml-1m/0.25/test.dat')
    b = Item_Based_CF(train)
    b.load_sim(df_train)
    b.item_cluster(df_train, 10)
    b.train()
    a.save(b, './data/ml-1m/0.1/trained_model')
    print("the trained model has been saved!")
    b.gen_all_ratings()
    b.test_from_all_ratings(test)
    b.top_n(test)
    pass