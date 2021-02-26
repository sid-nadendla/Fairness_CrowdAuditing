import numpy as np

def creditRules(ruleNo, creditScore, df):
    newLabel = np.zeros(len(df))
    if ruleNo == 0:
        creditHist = {0:5, 1:0, 2:3, 3:4, 4:4}
        employmentSince = {0:2, 1:3, 2:4, 3:5, 4:0}
        gender = {0:3, 1:5}
        marital = {0:1, 1:3, 2:5, 3:2}
        prop = {0:3, 1:4, 2:1, 3:5}
        employment = {0:4, 1:5, 2:2, 3:3}
        for i in range(0, len(df)-1):
            points = creditHist[df['Credit history'][i]] + employmentSince[df['Present employment since'][i]] + gender[df['Gender'][i]] + marital[df['Marital Status'][i]] + prop[df['Property'][i]] + employment[df['Employment'][i]]
            if points > 20: 
                newLabel[i] = 1
            else:
                newLabel[i] = 0
                    
    if ruleNo == 1:
        for i in range(0, len(df)-1):
            if df['Housing'][i] == 1 and df['Present employment since'][i] != 0 and df['Present employment since'][i] != 4:
                newLabel[i] = 1
            else:
                newLabel[i] = 0
    
    elif ruleNo == 2:
        for i in range(0, len(df)-1):
            if df['Credit history'][i] == 2 and df['Credit amount'][i] > 3000 and df['Housing'][i] == 1:
                newLabel[i] = 1
            else:
                newLabel[i] = 0 

    elif ruleNo == 3:
        for i in range(0, len(df)-1):
            if (df['Employment'][i] == 1 and df['Property'][i] == 3) or (df['Employment'][i] == 1 and df['Property'][i] == 1):
                newLabel[i] = 1
            else:
                newLabel[i] = 0        

    elif ruleNo == 4:
        for i in range(0, len(df)-1):
            if (df['Purpose'][i] == 1 or df['Purpose'][i] == 4) and df['Duration in months'][i] >= 24 and df['Credit amount'][i] >= 5000:
                newLabel[i] = 1
            elif df['Purpose'][i] != 1 and df['Credit amount'][i] >= 1500:
                newLabel[i] = 1
            else:
                newLabel[i] = 0

    elif ruleNo == 5:
        for i in range(0, len(df)-1):
            if (df['Present employment since'][i] != 4 or df['Present employment since'][i] != 0) and df['Purpose'][i] != 1:
                newLabel[i] = 1
            else:
                newLabel[i] = 0

    return newLabel