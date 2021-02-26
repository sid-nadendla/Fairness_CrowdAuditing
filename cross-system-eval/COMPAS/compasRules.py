import numpy as np

def compasRules(a, b, df):
    indices = [a, b]
    decile_score = []
    for index in indices:
        if df['c_charge_degree'][index] == 0:
            decile_score.append(7)
        elif df['priors_count'][index] <= 4:
            decile_score.append(4)
        elif df['priors_count'][index] > 4 and df['priors_count'][index] <= 6: 
            decile_score.append(6)         
        elif df['priors_count'][index] > 6 and df['priors_count'][index] <= 8:
            decile_score.append(8)
        elif df['priors_count'][index] > 8:
            decile_score.append(10)

    return decile_score
