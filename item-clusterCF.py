#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from data_manager import Data_Factory
from sklearn.cluster import KMeans
import pandas as pd

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
        pass

    def sim_cal(self, m1, m2):
        if m1 > m2:
            return self.sim_cal(m2, m1)
        self.similarity.setdefault(m1, {})
        sim = self.similarity.get(m1, {}).get(m2)
        if sim:
            return sim

        m1_user = self.movie_user.get(m1, {})
        m2_user = self.movie_user.get(m2, {})
        common_user = []
        for uid in m1_user:
            if uid in m2_user:
                common_user.append(uid)
        n = len(common_user)
        if n == 0:
            self.similarity[m1][m2] = 0
            return self.similarity[m1][m2]

        m1_rat = np.array([self.movie_user[m1][uid] for uid in common_user])
        m2_rat = np.array([self.movie_user[m2][uid] for uid in common_user])
        sum_m1 = np.sum(m1_rat)
        sum_m2 = np.sum(m2_rat)
        sum_inner = np.sum(m1_rat * m2_rat)
        sum_m1_square = np.sum(m1_rat ** 2)
        sum_m2_square = np.sum(m2_rat ** 2)
        denominator = np.sqrt((sum_m1_square - sum_m1 ** 2 / n) * (sum_m2_square - sum_m2 ** 2 / n))

        if denominator == 0:
            self.similarity[m1][m2] = 0
            return self.similarity[m1][m2]

        corr = (sum_inner - sum_m1 * sum_m2 / n) / denominator
        self.similarity[m1][m2] = corr * n / (n + 100)
        return self.similarity[m1][m2]

    def pred(self, user, movie, gama_list):
        item_list = self.user_movie.get(user, {})
        sim_acc = 0.0
        rat_acc = 0.0
        sim_acc_cluster = 0.0
        rat_acc_cluster = 0.0
        for item in item_list:
            if movie in self.labels.index and item in self.clusters[self.labels[movie]]:
                sim = self.sim_cal_cluster(item, movie)
                rat_acc_cluster += sim * self.user_movie[user][item]
                sim_acc_cluster += sim
            else:
                sim = self.sim_cal(item, movie)
                if sim <= 0:
                    continue
                rat_acc += sim * self.user_movie[user][item]
                sim_acc += sim
        pred_cf = 0.0
        pred_cluster = 0.0
        if sim_acc == 0:
            pred_cf = self.ave
        else:
            pred_cf = rat_acc / sim_acc
        if sim_acc_cluster == 0:
            pred_cluster = self.ave
        else:
            pred_cluster = rat_acc_cluster / sim_acc_cluster
        return [gama * pred_cluster + (1 - gama) * pred_cf for gama in gama_list]

    def test(self, test_rat, gama_list):
        test_rat = np.array(test_rat)
        n = test_rat.shape[0]
        err_square = [0.0 for x in range(len(gama_list))]
        for i in range(n):
            pred_rat = self.pred(test_rat[i][0], test_rat[i][1], gama_list)
            err_square = [err_square[x] + (pred_rat[x] - test_rat[i][2]) ** 2 for x in range(len(gama_list))]
            if i % 100 == 0:
                print ("processing items quantity: %d" % i)
        precise = [np.sqrt(err_square[x] / n) for x in range(len(gama_list))]
        print ("the rmse on test data is: ", precise)
        return precise

    def item_cluster(self, X, cluster_num):
        self.df_genome = X
        self.genome_shape = self.df_genome.shape
        kmeans = KMeans(n_clusters=cluster_num).fit(self.df_genome)
        self.labels = pd.Series(kmeans.labels_, index=self.df_genome.index)
        self.clusters = {}
        for label in range(cluster_num):
            self.clusters[label] = []
        for mid, label in self.labels.iteritems():
            self.clusters[label].append(mid)
        for label in self.labels:
            self.clusters[label] = set(self.clusters[label])
        return

    def sim_cal_cluster(self, m1, m2):
        if m1 > m2:
            return self.sim_cal_cluster(m2, m1)
        self.sim_cluster = {}
        if m1 in self.sim_cluster and m2 in self.sim_cluster[m1]:
            return self.sim_cluster[m1][m2]
        sim = 0.0
        diff = self.df_genome.loc[m1] - self.df_genome.loc[m2]
        sim = np.sqrt(np.sum(diff.values ** 2))
        self.sim_cluster.setdefault(m1, {})
        self.sim_cluster[m1][m2] = sim
        return self.sim_cluster[m1][m2]



if __name__ == '__main__':
    a = Data_Factory()
    df_train = a.generate_genome()
    """
    R = a.read_rating('./data/ml-1m/ratings.dat')
    train, valid, test = a.generate_train_valid_test_file(R, 0.002)
    a.save(train, './data/ml-1m/0.002/train.dat')
    a.save(test, './data/ml-1m/0.002/test.dat')
    """
    train = a.load('./data/ml-1m/0.002/train.dat')
    test = a.load('./data/ml-1m/0.002/test.dat')
    b = Item_Based_CF(train)
    res = {}
    gama_list = [0.2, 0.4, 0.5, 0.6, 0.8]
    for n_cluster in [5, 10, 15, 20, 25, 30]:
        b.item_cluster(df_train, n_cluster)
        print ('n_cluster: %d' % n_cluster)
        precise = b.test(test, gama_list)
        res[n_cluster] = precise
    pass