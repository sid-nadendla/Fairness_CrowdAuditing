from numpy.lib.shape_base import split
import scipy as sp
import pandas as pd
import numpy as np
import random
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import statistics
from itertools import product
from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot
import cvxpy as cp

import bbbb

def computezTildeGivenEx(df):
    sTilde = (df.groupby(['workclass','education','marital-status','occupation', 'audit-evals']).size() / len(df)).reset_index(name='sTilde')
    
    pr_sTildeGivenEx = sum(list(sTilde.loc[sTilde['audit-evals'] == 1]['sTilde']))

    yTilde = (df.groupby(['workclass','education','marital-status','occupation', 'income-prediction']).size() / len(df)).reset_index(name='yTilde')
    pr_yTilde1 = sum(list(yTilde.loc[yTilde['income-prediction'] == 1]['yTilde']))
    pr_yTilde0 = sum(list(yTilde.loc[yTilde['income-prediction'] == 0]['yTilde']))

    # print(pr_yTilde0, pr_yTilde1)
    
    return (pr_sTildeGivenEx - pr_yTilde1) / (pr_yTilde0 - pr_yTilde1)

if __name__ == "__main__":
    splits = 10
    adult = pd.read_csv('adult1.csv')
    split_adult = np.array_split(adult, splits)

    # grp = (split_adult[2].groupby(['workclass','education','marital-status','occupation']).size()).reset_index(name='count')    
    # print(max(grp['count']))
    compas = pd.read_csv('./COMPAS/compas-two-years-pre.csv')
    split_compas = np.array_split(compas, splits)    

    zGivenPi = []
    zTildeGivenEx = []
    for i in range(10):
        print('Partition ', i)
        # Compute P(z | pi)
        zGivenPi.append(bbbb.computezGivenPi(split_compas[i]))  
        
        # Compute P(z-tilde | x-tilde)
        grp = (split_adult[i].groupby(['workclass','education','marital-status','occupation', 'audit-evals', 'income-prediction']).size()).reset_index(name='count')
        zTildeGivenEx.append(computezTildeGivenEx(grp))

    print(zGivenPi, zTildeGivenEx)

    # zTildeGivenEx = np.array(zTildeGivenEx)
    # exTildeGivenPi = np.dot(zGivenPi, zTildeGivenEx.T)

    x = cp.Variable(splits)
    cost = cp.sum_squares(zTildeGivenEx @ x - zGivenPi)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    print('Final ', prob.value)

    # pr_yTildeGivenEx = (adult.groupby(['workclass','education','marital-status','occupation', 'income-prediction']).size() / len(adult)).reset_index(name='pr_yTildeGivenEx')
    # final = 1
    # for i in range(2):
    #     final *= sum(list(pr_yTildeGivenEx.loc[pr_yTildeGivenEx['income-prediction'] == i]['pr_yTildeGivenEx']))
    #     # print('Final ', final)
    # print(final*exTildeGivenPi)

    # print(pr_yTildeGivenEx)



    # b = inv(zTildeGivenEx.T.dot(zTildeGivenEx)).dot(zTildeGivenEx.T).dot(zGivenPi)
    # print(b)
    # yhat = zTildeGivenEx.dot(b)
    # # plot data and predictions
    # pyplot.scatter(zTildeGivenEx, zGivenPi)
    # pyplot.plot(zTildeGivenEx, yhat, color='red')
    # pyplot.show()