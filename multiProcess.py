#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file multiProcess.py
@author libo
@date 2017/12/4

This module provides a multiprocess implementing of calculating similarities by Ratings(Pearson Correlation Coefficient)
and Tag Genomes(Euclidean Distance)
"""

import numpy as np
from data_manager import Data_Factory
import time
from multiprocessing import Pool
import os

TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

def preprocess(train):
    """
    @brief: input Ratings, output movie_user dict and movie_id list
    """
    movie_user = {}
    data = np.array(train)
    for i in range(len(data)):
        uid, mid, rat = int(data[i][0]), int(data[i][1]), int(data[i][2])
        movie_user.setdefault(mid, {})
        movie_user[mid][uid] = rat
    return movie_user, list(movie_user.keys())


def sim_cal(m1, m2):
    """
    @brief: find common users of two movies, and calculate similarity between them using Pearson Correlation Coefficient
    """
    global movie_user
    sim = 0.0
    m1_user = movie_user.get(m1, {})
    m2_user = movie_user.get(m2, {})
    common_user = []
    for uid in m1_user:
        if uid in m2_user:
            common_user.append(uid)
    n = len(common_user)
    if n == 0:
        return sim

    m1_rat = np.array([movie_user[m1][uid] for uid in common_user])
    m2_rat = np.array([movie_user[m2][uid] for uid in common_user])
    sum_m1 = np.sum(m1_rat)
    sum_m2 = np.sum(m2_rat)
    sum_inner = np.sum(m1_rat * m2_rat)
    sum_m1_square = np.sum(m1_rat ** 2)
    sum_m2_square = np.sum(m2_rat ** 2)
    denominator = np.sqrt((sum_m1_square - sum_m1 ** 2 / n) * (sum_m2_square - sum_m2 ** 2 / n))

    if denominator == 0:
        return sim

    corr = (sum_inner - sum_m1 * sum_m2 / n) / denominator
    sim = corr * n / (n + 100)
    return sim

def sim_cluster(m1, m2):
    """
    @brief: calculate Euclidean Distance of two movies according to their Tag Genomes
    """
    global df_genome
    diff = df_genome.loc[m1] - df_genome.loc[m2]
    sim = np.sqrt(np.sum(diff.values ** 2))
    return sim

def childProcess(begin, end):
    """
    @brief: child process function of calculating similarities by Ratings
    """
    print("[%s]subprocess %s begin." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
    global mid_list
    sim_res = []
    for i in range(begin, end):
        for j in range(i + 1, len(mid_list)):
            sim = sim_cal(mid_list[i], mid_list[j])
            sim_res.append([mid_list[i], mid_list[j], sim])
    file = './data/ml-1m/subprocess/sim_cf_' + str(os.getpid())
    with open(file, 'w') as f:
        for i in range(len(sim_res)):
            f.write("%s,%s,%s" % (str(sim_res[i][0]), str(sim_res[i][1]), str(sim_res[i][2])))
    print("[%s]subprocess %s done." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
    return

def clusterProcess(begin, end):
    """
    @brief: child process function of calculating similarities by Tag Genomes
    """
    print("[%s]cluster subprocess %s begin." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
    global df_genome
    mid_list = list(df_genome.index)
    sim_res = []
    for i in range(begin, end):
        for j in range(i + 1, len(mid_list)):
            sim = sim_cluster(mid_list[i], mid_list[j])
            sim_res.append([mid_list[i], mid_list[j], sim])
            break
        break
    file = './data/ml-1m/subprocess/sim_cluster_' + str(os.getpid())
    with open(file, 'w') as f:
        for i in range(len(sim_res)):
            f.write("%s,%s,%s" % (str(sim_res[i][0]), str(sim_res[i][1]), str(sim_res[i][2])))
    print("[%s]subprocess %s done." % (time.strftime(TIMEFORMAT, time.localtime()), os.getpid()))
    return

def genProcess(k, n, type):
    """
    @brief: generate above two types of child processes, split calculating task evenly by indicating index range
    """
    index = [0]
    index = index + [n - 1 - int((-1 + np.sqrt(1 + 4*i*n*(n-1)/k)) / 2) for i in range(k - 1, 0, -1)]
    index.append(n - 1)
    p = Pool(k)
    for i in range(k):
        if type == 'cf':
            p.apply_async(childProcess, args=(index[i], index[i+1]))
        else:
            p.apply_async(clusterProcess, args=(index[i], index[i + 1]))
    print("[%s]Waiting for all subprocesses done..." % time.strftime(TIMEFORMAT, time.localtime()))
    p.close()
    p.join()
    print("[%s]All subprocesses done." % time.strftime(TIMEFORMAT, time.localtime()))

if __name__ == '__main__':
    a = Data_Factory()
    train = a.load('./data/ml-1m/0.1/train.dat')
    df_genome = a.generate_genome()
    movie_user, mid_list = preprocess(train)
    #calculate similarities according to Ratings using Pearson Correlation Coefficient
    genProcess(32, len(mid_list), 'cf')
    #calculate similarties according to Tag Genomes using Euclidean Distance
    genProcess(32, df_genome.shape[0], 'cluster')
    pass

