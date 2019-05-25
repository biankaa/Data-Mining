 ##Data Preprocessing Template

 ##Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#
import pandas as pd

data3=pd.read_csv('Data.csv')
X=data3.iloc[:,:-1].values
y=data3.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
###class
Labelencoder_X=LabelEncoder()
X[:,0]=Labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()
#print(X)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


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
          

## Checking Balance of Dataset
    
def check_balance(df, target):
    ##check = []
    print ('size of data is:', df.shape[0])
    for i in [0,1]:
        print("for target {} =".format(i))
        print(df[target].value_counts()[i]/df.shape[0]*100,"%")
        

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
