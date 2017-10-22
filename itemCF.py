#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from data_manager import Data_Factory

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

    def pred(self, user, movie):
        item_list = self.user_movie.get(user, {})
        sim_acc = 0.0
        rat_acc = 0.0
        for item in item_list:
            sim = self.sim_cal(item, movie)
            if sim <= 0:
                continue
            rat_acc += sim * self.user_movie[user][item]
            sim_acc += sim
        if sim_acc == 0:
            return self.ave
        return rat_acc / sim_acc

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





if __name__ == '__main__':
    a = Data_Factory()
    """
    R = a.read_rating('./data/ml-1m/ratings.dat')
    train, valid, test = a.generate_train_valid_test_file(R, 0.002)
    a.save(train, './data/ml-1m/0.002/train.dat')
    a.save(test, './data/ml-1m/0.002/test.dat')
    """
    train = a.load('./data/ml-1m/0.002/train.dat')
    test = a.load('./data/ml-1m/0.002/test.dat')
    b = Item_Based_CF(train)
    output = b.test(test)
    pass