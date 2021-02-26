import pandas as pd
import scipy as sp
import numpy as np
from sklearn import preprocessing
import math
import random
import statistics

"""
{'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp-inc': 3, 'Self-emp-not-inc': 4, 'State-gov': 5, 'Without-pay': 6}
{'10th': 0, '11th': 1, '12th': 2, '1st-4th': 3, '5th-6th': 4, '7th-8th': 5, '9th': 6, 'Assoc-acdm': 7, 'Assoc-voc': 8, 'Bachelors': 9, 'Doctorate': 10, 'HS-grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-school': 14, 'Some-college': 15}
{'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
{'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 
'Priv-house-serv': 8, 'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 'Tech-support': 12, 'Transport-moving': 13}
{'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4, 'hite': 5}
{'Female': 0, 'Male': 1}
{'<=50K': 0, '>50K': 1}
"""

df = pd.read_csv("adult.csv")

age = []
for i in range(len(df)):
    if df['age'][i] >= 17 and df['age'][i] <= 25:
        age.append(1)
    elif df['age'][i] > 25 and df['age'][i] <= 40:
        age.append(2)
    elif df['age'][i] > 40 and df['age'][i] <= 55:
        age.append(3)
    elif df['age'][i] > 55 and df['age'][i] <= 70:
        age.append(4)
    else:
        age.append(5)

df = df.drop('age', 1)
df['age'] = age

le = preprocessing.LabelEncoder()
for column in ['workclass','education','marital-status','occupation','race','sex','income-prediction']:
    le.fit(df[column])
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)
    df[column] = le.transform(df[column])

evals = [0]*len(df)
for i in range(len(df)):
    if df['occupation'][i] == 1 or df['occupation'][i] == 3 or df['occupation'][i] == 9 or df['occupation'][i] == 11 or df['occupation'][i] == 12:
        if df['income-prediction'][i] == 1:
            evals[i] = 1

    else:
        if df['income-prediction'][i] == 0:
            evals[i] = 1

df['audit-evals'] = evals

df.to_csv('adult1.csv')