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

kappa = 5.11

creditHist = {0:5, 1:0, 2:3, 3:4, 4:4}
employmentSince = {0:2, 1:3, 2:4, 3:5, 4:0}
gender = {0:3, 1:5}
marital = {0:1, 1:3, 2:5, 3:2}
prop = {0:3, 1:4, 2:1, 3:5}
employment = {0:4, 1:5, 2:2, 3:3}

def auditorOP(i, df):
    points = creditHist[df['Credit history'][i]] + employmentSince[df['Present employment since'][i]] + gender[df['Gender'][i]] + marital[df['Marital Status'][i]] + prop[df['Property'][i]] + employment[df['Employment'][i]]
    if points > 20: 
        return 1
    else:
        return 0

def hammingDist(i, j):
    return abs(i-j)

def validate(df):
    gLhsMean, gRhsMean = [], []
    for i in range(50):
        indexList = list(df.index)
        gLhs, gRhs = [], []
        for _ in range(len(df)):
            i, j = random.sample(indexList, 2)
            fDelta = hammingDist(df["Credit Risk"][i], df["Credit Risk"][j]) - kappa
            gLhs.append(hammingDist(auditorOP(i, df), auditorOP(j, df)))
            gRhs.append(hammingDist(auditorOP(i, df), df["Credit Risk"][i]) + hammingDist(auditorOP(j, df), df["Credit Risk"][j]) + kappa + fDelta)
            
            indexList = list(set(indexList) - set([i, j]))
            if len(indexList) < 2:
                break

        gLhsMean.append(statistics.mean(gLhs))
        gRhsMean.append(statistics.mean(gRhs))

    print("LHS Mean: ", str(statistics.mean(gLhsMean)))
    print("RHS Mean: ", str(statistics.mean(gRhsMean)))
    print("---------------------------")

if __name__ == "__main__":
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