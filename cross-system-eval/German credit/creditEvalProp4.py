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

creditHist = {0:5, 1:0, 2:3, 3:4, 4:4}
employmentSince = {0:2, 1:3, 2:4, 3:5, 4:0}
gender = {0:3, 1:5}
marital = {0:1, 1:3, 2:5, 3:2}
prop = {0:3, 1:4, 2:1, 3:5}
employment = {0:4, 1:5, 2:2, 3:3}

class StatisticalParity:
    def __init__(self, df):
        self.df = self.preprocess(df)
        self.sysGender_0, self.sysGender_1 = self.sysOutputGrouped()
        self.audGender_0, self.audGender_1 = self.audOutputGrouped()

    def preprocess(self, df):   
        le = preprocessing.LabelEncoder()
        for column in ['Credit history', 'Purpose' ,'Present employment since', 'Gender', 'Marital Status', 'Property' , 'Housing', 'Employment']:
            le.fit(df[column])
            df[column] = le.transform(df[column])

        return df

    def sysOutputGrouped(self):
        count_series = list(self.df.groupby(['Gender', 'Credit Risk']).size())
        gender_0 = count_series[:2]
        gender_1 = count_series[2:]

        return gender_0, gender_1 

    def audOutputGrouped(self):
        Aoutputs = []
        for i in range(len(self.df)):
            Aoutputs.append(self.auditorOutputs(i))

        dfCopy = self.df.copy()
        dfCopy = dfCopy.drop('Credit Risk', 1)
        dfCopy['auditor_output'] = Aoutputs

        a_count_series = dfCopy.groupby(['Gender', 'auditor_output']).size()
        print('aud', a_count_series)
        aud_race_0 = a_count_series[:2]
        aud_race_1 = a_count_series[2:]

        return aud_race_0, aud_race_1

    def auditorOutputs(self, i):
        points = creditHist[self.df['Credit history'][i]] + employmentSince[self.df['Present employment since'][i]] + gender[self.df['Gender'][i]] + marital[self.df['Marital Status'][i]] + prop[self.df['Property'][i]] + employment[self.df['Employment'][i]]
        if points > 20: 
            return 1
        else:
            return 0

    def computeProbs(self, list1, list2, flag):
        # print(list1, list2)
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

    def validate(self):
        epsilon = 1

        M1 = self.computeProbs(self.sysGender_0, self.audGender_0, 0) / epsilon
        M2 = self.computeProbs(self.sysGender_1, self.audGender_1, 1) / epsilon
        delta = self.computeProbs(self.sysGender_0, self.sysGender_1, 2)

        print(M1, M2) #---------------> 
        print(delta) #---------------> 

        LHS = self.computeProbs(self.audGender_0, self.audGender_1, 2)
        RHS = (M1*epsilon) + (M2*epsilon) + delta

        print(LHS, RHS) #--------------> 


df = pd.read_csv("Credit Risk.csv")
# count_series = list(df.groupby(['Gender', 'Credit Risk']).size())
# print(count_series)
stat = StatisticalParity(df)
stat.validate()