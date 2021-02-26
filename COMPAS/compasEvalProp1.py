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

kappa = 4
epsilon = 4

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

def mahalanobis(i, j, df):
    cov = np.cov(df.values.T)
    inv_covmat = sp.linalg.inv(cov)

    diff = [x-y for x, y in zip(list(df.loc[i]), list(df.loc[j]))]
    temp = np.matmul(diff, inv_covmat)
    maha = [x*y for x, y in zip(diff, temp)]
    return math.sqrt(sum(maha))

def validate(df):
    gLhsMean, gRhsMean = [], []
    for i in range(5):
        indexList = list(df.index)
        gLhs, gRhs = [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            sysDist = hammingDist(df["decile_score"][i], df["decile_score"][j])
            similarity = mahalanobis(i, j, df)

            if similarity < kappa:
                dist1 = hammingDist(auditorOP(i, df), auditorOP(j, df))
                # dist2 = hammingDist(auditorOP(j, df), df["decile_score"][j])
                print(dist1)

            indexList = list(set(indexList) - set([i, j]))
            if len(indexList) < 2:
                break

        gLhsMean.append(statistics.mean(gLhs))
        gRhsMean.append(statistics.mean(gRhs))
    
    # outputFile.write("LHS Mean: " + str(statistics.mean(gLhsMean)) + "\n")
    # outputFile.write("RHS Mean: " + str(statistics.mean(gRhsMean)) + "\n") 
    # outputFile.write("----------------------------\n")

    print("LHS Mean: ", str(statistics.mean(gLhsMean)))
    print("RHS Mean: ", str(statistics.mean(gRhsMean)))
    print("---------------------------")

    # plt.plot(gLhsMean, label="LHS")
    # plt.plot(gRhsMean, label="RHS")
    # plt.legend()
    # plt.show()

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