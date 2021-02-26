import pandas as pd
import scipy as sp
import numpy as np
from sklearn import preprocessing
import math
import random
import statistics

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
 
def mahalanobis(df):
    x_minus_mu = df[:] - np.mean(df)
    cov = np.cov(df.values.T)
    inv_covmat = sp.linalg.inv(cov)

    distMean = []
    for _ in range(100):
        indexList = list(df.index)
        dist = []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)  
            diff = [x-y for x, y in zip(list(df.loc[i]), list(df.loc[j]))]
            temp = np.matmul(diff, inv_covmat)
            maha = [x*y for x, y in zip(diff, temp)]
            dist.append(math.sqrt(sum(maha)))

    print(min(dist), max(dist))

def intervals(parts, duration):
    part_duration = duration / parts
    return [(i * part_duration, (i + 1) * part_duration) for i in range(parts)]

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
    return bin(i^j).count('1')

def delta(df):
    distMean = []
    for i in range(1):
        indexList = list(df.index)
        dist1, dist2 = [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            # dist1.append(hammingDist(auditorOP(i, df), auditorOP(j, df)))
            # dist2.append(hammingDist(df['decile_score'][i], df['decile_score'][j]))
            dist1.append(hammingDist(auditorOP(i, df), df['decile_score'][i]))
            dist2.append(hammingDist(auditorOP(j, df), df['decile_score'][j]))
        print(max(dist1), max(dist2))

if __name__ == "__main__":
    # print(intervals(10, 8.62))
    df = pd.read_csv("compas-scores-two-years.csv")
    le = preprocessing.LabelEncoder()
    for column in ['sex', 'race', 'c_charge_degree']:
        le.fit(df[column])
        df[column] = le.transform(df[column])


    # delta(df)

    # df = pd.read_csv("Credit Risk.csv")
    # le = preprocessing.LabelEncoder()
    # for column in ['Credit history', 'Purpose', 'Present employment since', 'Gender', 'Marital Status', 'Property' , 'Housing', 'Employment']:
    #     le.fit(df[column])
    #     df[column] = le.transform(df[column])

    cores = mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0)
    pool = Pool(cores)
    df_out = np.vstack(pool.map(delta, df_split))
    pool.close()
    pool.join()
    pool.clear()

# # print(max(distMean)) #-------------------- 3.6159545438342917
# print(statistics.mean(distMean)) #-------- 3.5926961198624165
# print(min(distMean)) #-------------------- 3.5603380586268565
  
#==============================DELTA=======================================

# kappa = 5.11
# deltaMean = []
# for _ in range(50):
#     indexList = list(df.index)
#     delta = []
#     for _ in range(len(df)):
#         i, j = random.sample(indexList, 2)
#         delta.append( abs(df["decile_score"][i] - df["decile_score"][j]) )
#     deltaMean.append(statistics.mean(delta))

# print(max(deltaMean))
# print(statistics.mean(deltaMean))
# print(min(deltaMean)) #---------- -0.371