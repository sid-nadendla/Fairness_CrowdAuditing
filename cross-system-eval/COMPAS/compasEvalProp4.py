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

class StatisticalParity:
    def __init__(self, df):
        self.df = df
        self.sysGender_0, self.sysGender_1 = self.sysOutputGrouped()
        self.audGender_0, self.audGender_1 = self.audOutputGrouped()

    def sysOutputGrouped(self):
        count_series = self.df.groupby(['sex', 'decile_score']).size()
        gender_0 = count_series[:10]
        gender_1 = count_series[10:]

        return gender_0, gender_1 

    def audOutputGrouped(self):
        Aoutputs = []
        for i in range(len(self.df)):
            Aoutputs.append(self.auditorOutputs(i))

        dfCopy = self.df.copy()
        dfCopy = dfCopy.drop('decile_score', 1)
        dfCopy['auditor_output'] = Aoutputs

        a_count_series = list(dfCopy.groupby(['sex', 'auditor_output']).size())
        aud_race_0 = a_count_series[:5]
        aud_race_1 = a_count_series[5:]

        return aud_race_0, aud_race_1

    def preprocess(self):
        le = preprocessing.LabelEncoder()
        for column in ['sex', 'race', 'c_charge_degree']:
            le.fit(self.df[column])
            self.df[column] = le.transform(self.df[column])

    def auditorOutputs(self, i):
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

    def computeProbs(self, list1, list2, flag):
        res = []
        if flag == 0:
            for i, j in zip(list1, list2):
                res.append( abs( (i/sum(self.sysGender_0)) - (j/sum(self.sysGender_0)) ))
        elif flag == 1:
            for i, j in zip(list1, list2):
                res.append( abs( (i/sum(self.sysGender_1)) - (j/sum(self.sysGender_1)) ))
        else:
            for i, j in zip(list1, list2):
                res.append( abs( (i/sum(self.sysGender_0)) - (j/sum(self.sysGender_1)) ))
        
        return max(res)

    # def computeEpsilon(self):
    #     epsilonMean = []
    #     for _ in range(50):
    #         epsilon = []
    #         for i in range(len(self.df)):
    #             epsilon.append(abs(self.auditorOutputs(i) - self.df['decile_score'][i]))
    #         epsilonMean.append(statistics.mean(epsilon))

    #     return statistics.mean(epsilonMean) #----------> 3

    def validate(self):
        epsilon = 3

        M1 = self.computeProbs(self.sysGender_0, self.audGender_0, 0) / epsilon
        M2 = self.computeProbs(self.sysGender_1, self.audGender_1, 1) / epsilon
        delta = self.computeProbs(self.sysGender_0, self.sysGender_1, 2)

        print(M1, M2) #---------------> 0.9627240143369175, 0.037520765309045086
        print(delta) #---------------> 0.030423633862867963

        LHS = self.computeProbs(self.audGender_0, self.audGender_1, 2)
        RHS = (M1*epsilon) + (M2*epsilon) + delta

        print(LHS, RHS) #--------------> 0.7573995950726239, 3.0311579728007554


df = pd.read_csv("compas-scores-two-years.csv")
count_series = df.groupby(['race', 'decile_score']).size()
print(count_series)

# stat = StatisticalParity(df)
# stat.preprocess()
# stat.validate()