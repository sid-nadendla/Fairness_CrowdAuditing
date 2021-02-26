import scipy as sp
import pandas as pd
import numpy as np
import random
import math
import statistics

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp

sensitive = ['sex', 'race', 'age_cat']
omega = 0.2

"""
---------------------------------------RULES--------------------------------------------
If an individual commits misdemeanor(1) and has prior count > 7, then YES(1). Else, NO(0)
If Felony, YES(1)
"""
def compasRule(df):
    is_recid = []
    for i in range(len(df)):
        if df['c_charge_degree'][i] == 1:
            if df['priors_count'][i] != 1 and df['priors_count'][i] != 2:
                is_recid.append(1)
            else:
                is_recid.append(0)
        else:
            is_recid.append(1)

    return is_recid

def auditEvalsCompas(df):
    evals = []
    for _, i in df.iterrows():
        if i['c_charge_degree'] == 0:
            if i['is_recid'] == 0:
                evals.append(0)
            else: 
                evals.append(1)

        else:
            if i['is_recid'] == 1:
                evals.append(0)
            else: 
                evals.append(1)
    return evals

"""
------------------------------------------------------------------------------------------
"""

def computeEta(df):
    eta = {}
    features = list(set(list(df)) - set(sensitive))
    for pi in sensitive:
        for x in features:
            eta_piX = 0
            pr_piGivenx = (df.groupby([pi, x]).size() / df.groupby(x).size()).reset_index(name='pr_piGivenx')
            pr_pi = (df.groupby(pi).size() / len(df)).reset_index(name='pr_pi')

            # print(pr_piGivenx)
            # print(pr_pi)
            # print('-------------------------------')
            
            for ppi in pr_piGivenx[pi].unique():
                eta_piX = sum(list(pr_piGivenx.loc[pr_piGivenx[pi] == ppi]['pr_piGivenx'])) / pr_pi.loc[pr_pi[pi] == ppi, 'pr_pi'].iloc[0]

            # print(eta_piX)
            if eta_piX < 1 - omega:
                if pi not in eta:
                    eta[pi] = [x]
                else:
                    eta[pi].append(x) 
    return eta

def computeExGivenPi(df):
    num, denom = 1, 1
    eta = computeEta(df)
    
    for pi in eta.keys():
        #Numerator
        temp1 = eta[pi] + [pi]
        grp1 = (df.groupby(temp1).size() / len(df)).reset_index(name='grp1')
        summ1 = 1
        for i in df[pi].unique():
            summ1 *= sum(list(grp1.loc[grp1[pi] == i]['grp1']))
        num *= summ1

        #Denominator
        features = list(set(list(df)) - set(sensitive))
        temp2 = features + [pi]
        grp2 = (df.groupby(temp2).size() / len(df)).reset_index(name='grp2')
        summ2 = 0
        for i in df[pi].unique():
            summ2 *= sum(list(grp2.loc[grp2[pi] == i]['grp2']))
        denom += summ2

    # print(num,denom)
    return num / denom

def computezGivenPi(df):
    audit_evals = auditEvalsCompas(df)
    df['auditor_evals'] = audit_evals
    df = df.drop(['is_recid', 'decile_score'], 1)
    # computeEta(df)

    features = list(set(list(df)) - set(sensitive))
    grp = (df.groupby(features).size() / len(df)).reset_index(name='pr_zGivenEx')
    pr_zGivenEx = sum(list(grp.loc[grp['auditor_evals'] == 1]['pr_zGivenEx']))

    df = df.drop('auditor_evals', 1)
    return pr_zGivenEx * computeExGivenPi(df)

# df = pd.read_csv('compas-two-years-pre.csv')
# computezGivenPi(df)