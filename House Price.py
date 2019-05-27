# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:02:28 2019

@author: 86753
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from scipy import stats
from scipy.stats import norm

df_train = pd.read_csv("C:/Users/86753/Downloads/train.csv")
df_test = pd.read_csv("C:/Users/86753/Downloads/test.csv")

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis =1, inplace = True)
df_train.shape

df_train.drop("Id", axis = 1, inplace = True)

df_train.describe()
df_train.describe(include = ["O"])

###Outliers
fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.show()

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

##Target Variable
sns.distplot(df_train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit (df_train['SalePrice'])
print('\n mu={:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()         

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(df_train['SalePrice'])
print('\n mu = {:2f} and sigma = {:2f}\n'.format(mu, sigma))
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot= plt)
plt.show()



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
        
a = check_missing_data(df_train).transpose()
a0= a[a['Percent'] > 0].sort_values('Percent',ascending=False)

all_data['PoolQC'] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrType"].fillna(0)
all_data['MSZoning'] = all_data["MSZoning"].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'],axis = 1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data["Electrical"] = all_data["Electrical"].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")




    