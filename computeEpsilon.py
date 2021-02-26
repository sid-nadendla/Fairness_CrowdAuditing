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

creditHist = {0:5, 1:0, 2:3, 3:4, 4:4}
employmentSince = {0:2, 1:3, 2:4, 3:5, 4:0}
gender = {0:3, 1:5}
marital = {0:1, 1:3, 2:5, 3:2}
prop = {0:3, 1:4, 2:1, 3:5}
employment = {0:4, 1:5, 2:2, 3:3}

# def auditorOP(i, df):
#     if df['c_charge_degree'][i] == 0:
#         return 7
#     if df['priors_count'][i] <= 4:
#         return 4
#     if df['priors_count'][i] > 4 and df['priors_count'][i] <= 6: 
#         return 6         
#     if df['priors_count'][i] > 6 and df['priors_count'][i] <= 8:
#         return 8
#     if df['priors_count'][i] > 8:
#         return 10

def auditorOP(i, df):
    # points = creditHist[df['Credit history'][i]] + employmentSince[df['Present employment since'][i]] + gender[df['Gender'][i]] + marital[df['Marital Status'][i]] + prop[df['Property'][i]] + employment[df['Employment'][i]]
    # if points > 20: 
    #     return 1
    # else:
    #     return 0

    if df['Credit history'][i] == 2 and df['Credit amount'][i] > 3000 and df['Housing'][i] == 1:
        return 1
    else:
        return 0

def validate(df):
    epMean = []
    for i in range(100):
        indexList = list(df.index)
        epsilon = []
        for _ in range(len(df)):
            i = random.choice(indexList)
            epsilon.append( abs(df['Credit Risk'][i] - auditorOP(i, df)) )
        epMean.append(max(epsilon))

    print(statistics.mean(epMean))

if __name__ == "__main__":
    # df = pd.read_csv("compas-scores-two-years.csv")
    # le = preprocessing.LabelEncoder()
    # for column in ['sex', 'race', 'c_charge_degree']:
    #     le.fit(df[column])
    #     df[column] = le.transform(df[column])

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