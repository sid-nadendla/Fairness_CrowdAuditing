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

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
 
kappa = 3.56

def auditorOP(i, df):
    if df['race'][i] == 0:
        return 3
    if df['race'][i] == 1:
        return 7
    if df['race'][i] == 2:
        return 4
    if df['race'][i] == 3:
        return 5
    if df['race'][i] == 4:
        return 3
    if df['race'][i] == 5:
        return 5

def hammingDist(i, j):
    # assert len(i) == len(j)
    # return sum(c1 != c2 for c1, c2 in zip(i, j))
    return abs(i-j)


def validate(df):
    gLhsMean, gRhsMean = [], []
    for i in range(7000):
        indexList = list(df.index)
        gLhs, gRhs, epsilon, fDelta = [], [], [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            fDelta = hammingDist(df["decile_score"][i], df["decile_score"][j]) - kappa
            
            ncfDist1 = hammingDist(auditorOP(i, df), df["decile_score"][i])
            ncfDist2 = hammingDist(auditorOP(j, df), df["decile_score"][j])

            # pi.append(ncfDist1) if ncfDist1 > 5 else epsilon.append(ncfDist1)
            # pi.append(ncfDist2) if ncfDist2 > 5 else epsilon.append(ncfDist2)
            gLhs.append(hammingDist(auditorOP(i, df), auditorOP(j, df)))
            # if ncfDist1 <= 5 and ncfDist2 <= 5:
            #     gRhs.append(ncfDist1 + ncfDist2 + kappa +  fDelta)

            # elif ncfDist1 > 5 and ncfDist2 > 5:
            #     gRhs.append(10.0 + kappa +  fDelta)
            
            if ncfDist1 > 5 or ncfDist2 > 5:
                epsilon = ncfDist1 if ncfDist1 <= 5 else ncfDist2
                gRhs.append(5.0 - (epsilon + kappa + fDelta))

            indexList = list(set(indexList) - set([i, j]))
            if len(indexList) < 2:
                break

        gLhsMean.append(statistics.mean(gLhs))
        gRhsMean.append(statistics.mean(gRhs))
    
    print("LHS Mean: ", statistics.mean(gLhsMean))
    print("RHS Mean: ", statistics.mean(gRhsMean))
    print("------------------------")

    # plt.plot(gLhsMean, label="LHS")
    # plt.plot(gRhsMean, label="RHS")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
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


# epsilon = []
# count = 0
# for _ in range(len(df)):
#     indexList = list(df.index)
#     i = random.sample(indexList, 1)[0]
#     # epsilon.append(hammingDist(auditorOP(i), df["decile_score"][i]))
#     temp = hammingDist(auditorOP(i), df["decile_score"][i])
#     if temp >= 6:
#         count += 1

# print(count)

# print(max(epsilon))
# print(statistics.mean(epsilon))
# print(min(epsilon))