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
        self.sysDict = {}
        self.auditorAbs = self.audOutputGrouped()

    def sysOutputGrouped(self):
        count_series = (self.df.groupby(['race', 'decile_score']).size() / self.df.groupby('race').size()).to_frame('probability').reset_index()
        for i in range(6):
            self.sysDict[i] = list(count_series.loc[count_series['race'] == i, 'probability'])

        # for i in self.sysDict:
        #     for j in list(set(self.sysDict) - set(i)):

        # return max(absDiff)

    def audOutputGrouped(self):
        a_count_series = (self.df.groupby(['sex', 'auditor_evals']).size() / self.df.groupby('sex').size()).to_frame('probability')
        aud_race_0 = list(a_count_series['probability'][:7])
        aud_race_1 = list(a_count_series['probability'][7:])

        absDiff = []
        for i, j in zip(aud_race_0, aud_race_1):
            absDiff.append(abs(i-j))
        
        return max(absDiff)


df = pd.read_csv("compas-two-years-pre.csv")
# count_series = df.groupby(['race', 'decile_score']).size()
# print(count_series)

stat = StatisticalParity(df)
stat.sysOutputGrouped()
# stat.audOutputGrouped()
# stat.preprocess()
# stat.validate()