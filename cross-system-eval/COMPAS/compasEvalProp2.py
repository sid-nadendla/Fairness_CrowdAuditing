## COMPAS data - Sensitive Attributes Based Metric.
import scipy as sp
import pandas as pd
import numpy as np
import random
import math
from sklearn import preprocessing
from scipy.spatial import distance
import matplotlib.pyplot as plt
import statistics

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

kappa = 3.56
delta = -0.371

def auditorOP(i, df):
    if df['c_charge_degree'][i] == 0:
        return 7
    if df['priors_count'][i] <= 4:
        return 4
    if df['priors_count'][i] > 4 and df['priors_count'][i] <= 6: 
        return 6         
    if df['priors_count'][i] > 6 and df['priors_count'][i] <= 8:
        return 8
    if df['priors_count'][i] > 8:
        return 10

def hammingDist(i, j):
    return abs(i-j)

def validate(df):
    gLhsMean, gRhsMean = [], []
    for i in range(7000):
        indexList = list(df.index)
        gLhs, gRhs = [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            if hammingDist(df["decile_score"][i], df["decile_score"][j]) > kappa + delta:
                gRhs.append(kappa + delta - (hammingDist(auditorOP(i, df), df["decile_score"][i]) + hammingDist(auditorOP(j, df), df["decile_score"][j])))

            gLhs.append(hammingDist(auditorOP(i, df), auditorOP(j, df)))
            
            indexList = list(set(indexList) - set([i, j]))
            if len(indexList) < 2:
                break

        gLhsMean.append(statistics.mean(gLhs))
        gRhsMean.append(statistics.mean(gRhs))

    print("LHS Mean: ", str(statistics.mean(gLhsMean)))
    print("RHS Mean: ", str(statistics.mean(gRhsMean)))
    print("---------------------------")

if __name__ == "__main__":
    df = pd.read_csv("compas-scores-two-years.csv")
    le = preprocessing.LabelEncoder()
    for column in ['sex', 'race', 'c_charge_degree']:
        le.fit(df[column])
        df[column] = le.transform(df[column])

    cores = mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0)
    pool = Pool(cores)
    df_out = np.vstack(pool.map(validate, df_split))
    pool.close()
    pool.join()
    pool.clear()