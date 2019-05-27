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
from scipy.stats import norm, skew

df_train = pd.read_csv("C:/Users/86753/Downloads/train.csv")
df_test = pd.read_csv("C:/Users/86753/Downloads/test.csv")



df_train.describe()
df_train.describe(include = ["O"])

###Outliers
fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.show()

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis =1, inplace = True)
all_data.drop("Id", axis = 1, inplace = True)

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
        
a = check_missing_data(all_data).transpose()
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


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu','BsmtQual','BsmtCond','GarageQual','GarageCond', 'ExterQual',
        'ExterCond', 'HeatingQC', 'PoolQC','KitchenQual', 'BsmtFinType1', 'BsmtFinType2',
        'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
        'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold',
        'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(all_data[c].values)
    all_data[c] = lbl.transform(all_data[c].values)
    
print('Shape all_data: {}'.format(all_data.shape))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data = pd.get_dummies(all_data)
ntrain = len(df_train)
train = all_data[:ntrain]
test = all_data[ntrain:]
y_train = df_train.SalePrice.values

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
##from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
##from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
##import lightgbm as lgb

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score1 = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))

score2 = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
