# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Data Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/overview
"""

import numpy as np
import pandas as pd
import os

print(os.listdir(r"C:\Users\86753\Downloads\santander-customer-transaction-prediction"))

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv(r"C:\Users\86753\Downloads\santander-customer-transaction-prediction\train.csv")
df_test = pd.read_csv(r"C:\Users\86753\Downloads\santander-customer-transaction-prediction\test.csv")

## EDA

df_train["target"].value_counts().plot.bar()


t0 = df_train[df_train["target"] == 0].var_0.plot.hist()
t0.set_title("target = 0")
t1 = df_train[df_train["target"] == 1].var_0.plot.hist()
t1.set_title("target = 1")
plt.show()

df_train["var_80"].hist()

## Checking Missing Value
def check_missing_data(df):
    flag = df.isna().sum().any()
    if flag == True:
       total = df.isnull().sum()
       percent = (df.isnull().sum())/(df.isnull().count()*100)
       output = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
       data_type = []
       for col in df.columns:
          dtype = str(df[col].dtype)
          data_type.append(dtype)
       output["Types"] = data_type
       return(np.transpose(output))
    else:
        return(False)
        
check_missing_data(df_train)  
check_missing_data(df_test)    

## Checking Balance of Dataset
    
def check_balance(df, target):
    ##check = []
    print ('size of data is:', df.shape[0])
    for i in [0,1]:
        print("for target {} =".format(i))
        print(df[target].value_counts()[i]/df.shape[0]*100,"%")
        
check_balance(df_train, "target")

print("Skewness: %f" % df_train["target"].skew())
print("Kurtosis: %f" % df_train["target"].kurt())

col_drop = ["target", "ID_code"]
X_train = df_train.drop(col_drop, axis = 1)
y_train = df_train["target"]
X_test  = test.drop("ID_code",axis=1)

train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state = 1)
rfc_model = RandomForestClassifier(random_state = 0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rfc_model, random_state = 1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)

y_pred_rfc = rfc_model.predict(X_test)

