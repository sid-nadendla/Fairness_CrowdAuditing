import pandas as pd
import scipy as sp
import numpy as np
from sklearn import preprocessing
import math
import random
import statistics

creditHist = {0:5, 1:0, 2:3, 3:4, 4:4}
employmentSince = {0:2, 1:3, 2:4, 3:5, 4:0}
gender = {0:3, 1:5}
marital = {0:1, 1:3, 2:5, 3:2}
prop = {0:3, 1:4, 2:1, 3:5}
employment = {0:4, 1:5, 2:2, 3:3}

def auditorOP(df):
    evals = []
    for i in range(len(df)):
        points = creditHist[df['Credit history'][i]] + employmentSince[df['Present employment since'][i]] + gender[df['Gender'][i]] + marital[df['Marital Status'][i]] + prop[df['Property'][i]] + employment[df['Employment'][i]]
        if points > 20: 
            if df['Credit Risk'][i] == 1:
                evals.append(1)
            else:
                evals.append(0)
        else:
            evals.append(0)

    return evals

def main():
    df = pd.read_csv("Credit Risk.csv")
    le = preprocessing.LabelEncoder()
    for column in ['Credit history', 'Purpose' ,'Present employment since', 'Gender', 'Marital Status', 'Property' , 'Housing', 'Employment']:
        le.fit(df[column])
        df[column] = le.transform(df[column])

    df['Auditor Evals'] = auditorOP(df)


    

    df.to_csv('credist-risk-pre.csv')

main()