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
 
kappa = 5.11
delta = -4.718

def auditorOP(i, df):
    if df['Credit history'][i] == 2 and df['Credit amount'][i] > 3000 and df['Housing'][i] == 1:
        return 1
    else:
        return 0

def hammingDist(i, j):
    return abs(i-j)

def validate(df):
    gLhsMean, gRhsMean = [], []
    for i in range(50):
        indexList = list(df.index)
        gLhs, gRhs, epsilon, fDelta = [], [], [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            fDelta = hammingDist(df["Credit Risk"][i], df["Credit Risk"][j]) - kappa
            
            ncfDist1 = hammingDist(auditorOP(i, df), df["Credit Risk"][i])
            ncfDist2 = hammingDist(auditorOP(j, df), df["Credit Risk"][j])

            gLhs.append(hammingDist(auditorOP(i, df), auditorOP(j, df)))
            
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

if __name__ == '__main__':
    df = pd.read_csv("Credit Risk.csv")
    le = preprocessing.LabelEncoder()
    for column in ['Credit history', 'Purpose' ,'Present employment since', 'Gender', 'Marital Status', 'Property' , 'Housing', 'Employment']:
        le.fit(df[column])
        df[column] = le.transform(df[column])

    cores = mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0)
    pool = Pool(cores)
    df_out = np.vstack(pool.map(validate, df_split))
    pool.close()
    pool.join()
    pool.clear()