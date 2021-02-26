import pandas as pd
import scipy as sp
import numpy as np
from sklearn import preprocessing
import math
import random
import statistics

df = pd.read_csv("compas-scores-two-years.csv")
le = preprocessing.LabelEncoder()
for column in ['sex', 'race', 'c_charge_degree', 'age_cat']:
    le.fit(df[column])
    df[column] = le.transform(df[column])

#----------------Encoding priors_count--------------------
n_prior = []
for i in df['priors_count']:
    if i >= 0 and i <= 3:
        n_prior.append(1)
    elif i >= 4 and i <= 7:
        n_prior.append(2)
    elif i >= 8 and i < 11:
        n_prior.append(3)
    elif i >=12 and i <= 15:
        n_prior.append(4)
    elif i >=16 and i<=19:
        n_prior.append(5)
    elif i >=20 and i<=23:
        n_prior.append(6)
    else:
        n_prior.append(7)

df = df.drop('priors_count', 1)
df['priors_count'] = n_prior
# df.to_csv('compas-two-years-pre2.csv')


# print(len(n_age), len(df['age']))

#---------------Appending auditor evals-----------------
# audOp = []
# for i in range(len(df)):
#     if df['c_charge_degree'][i] == 0:
#         audOp.append(7)
#     elif df['priors_count'][i] == '0-3':
#         audOp.append(2)
#     elif df['priors_count'][i] == '4-7': 
#         audOp.append(4)
#     elif df['priors_count'][i] == '8-11': 
#         audOp.append(5)
#     elif df['priors_count'][i] == '12-15': 
#         audOp.append(6)         
#     elif df['priors_count'][i] == '16-19': 
#         audOp.append(7)
#     elif df['priors_count'][i] == '20-23': 
#         audOp.append(8)
#     elif df['priors_count'][i] == '>23': 
#         audOp.append(10)

# df['auditor_evals'] = audOp
df.to_csv("compas-two-years-pre.csv", index=False)