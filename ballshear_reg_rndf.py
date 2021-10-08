# -*- coding: utf-8 -*-
"""
Created on Thu 8-Oct-2021

Link :
    Rnd Forestregression model : https://levelup.gitconnected.com/random-forest-regression-209c0f354c84
    Onehot encoder : https://towardsdatascience.com/columntransformer-in-scikit-for-labelencoding-and-onehotencoding-in-machine-learning-c6255952731b
                     https://stackoverflow.com/questions/58087238/valueerror-setting-an-array-element-with-a-sequence-when-using-onehotencoder-on

@author: 41162395
"""


# from numpy import array
# from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from sklearn.ensemble import IsolationForest
#from sklearn.preprocessing import MinMaxScaler

# Import raw dataset to pandas dataframe
df_org = pd.read_csv('ballshear.csv')
df = pd.read_csv('ballshear.csv')

# Data cleasing drop NAN columns
to_drop = ['LSL','USL','Parameter.Recipe','PROJECT_TYPE']
df_org.drop(to_drop, inplace=True, axis=1)
df_org.dropna(inplace=True)
df.drop(to_drop, inplace=True, axis=1)
df.dropna(inplace=True)
df_org.dropna(inplace=False)

# Drop unwant columns
to_drop = ['C_RISTIC','DATE_TIME','SHIFT','PT','EN_NO','DEVICE','REMARK','SD',
           'BOM_NO','SUBGRP','PLANT_ID','MC_ID','MC_NO','COUNTER',
            'CHAR_MINOR','PACKAGE','DATE_','CIMprofile.cim_machine_name',
            'Parameter.DataType','Parameter.Unit',
            'Parameter.Valid','Parameter.EquipOpn','Parameter.EquipID',
            'Parameter.ULotID','Parameter.CreateTime']
df.drop(to_drop, inplace=True, axis=1)

# Rearrange column position
cols_to_move = ['Parameter.Max','Parameter.Min','Parameter.Value','MEANX']
new_cols = np.hstack((df.columns.difference(cols_to_move), cols_to_move))
df = df.reindex(columns=new_cols)

# Input json for prediction 
#dfx = pd.read_json('x_input.json')  

# One hot encoder to numpy array X as below 9 columns or features
# Parameter.BondType,Parameter.Group,Parameter.No,Parameter.No_1,WIRE_SIZE,Parameter.Max,Parameter.Min,Parameter.Value, MEANX
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [0,1,2,3,4,5])], remainder='passthrough')
X = ct.fit_transform(df).toarray()  # upper case 'X'
#X0 = ct.transform(dfx).toarray()  # upper case 'X0'

# Split X to 2 arrays as x inputFeatures and y outputResponse
selector = [i for i in range(X.shape[1]) if i != 47] #Column 35 is 'MEANX' 
x = X[:,selector]  # lower case 'x'  all column except 'MEANX'
y = X[:,47]  # lower case 'y'  for 'MEANX'

# Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Random Forest Regressor Algorithm 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=(0))
regressor.fit(X_train, y_train)

# Predicting the results of the Test set
y_pred = regressor.predict(X_test)
#y_pred_2 = np.ravel(y_pred)  # Reduce shape of y_pred
compare = pd.DataFrame(y_test)
compare['y_pred'] = y_pred.tolist()
compare.to_csv('compare.csv')

# printing the results of the current iteration
np.set_printoptions(precision=2)
compare = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
print(compare)
MAPE = np.mean(100 * (np.abs(y_test-regressor.predict(X_test))/y_test))
print('Accuracy:', 100-MAPE)

#add 'Y' array as new column in df DataFrame
PRED = regressor.predict(x)
PRED_2 = np.ravel(PRED)  # Reduce shape of y_pred
df['PRED'] = PRED_2.tolist()
#Export output to csv
df.to_csv('ballshear_regression.csv')


#############################
# Common command 
#-------------------------
# df.describe
# df.info()
# df.index
# df.columns
# dfx[:5]
# du1=ds1.unstack()
# df.isna().sum()
# dfx = df.set_index(['C_RISTIC','PT','Parameter.CreateTime','Parameter.BondType','Parameter.No'])
# df.loc[25024]
# for i in range (len(Y)):
#     if Y[i] == -1:
#         print (df.loc[i])
#
# selected_columns = df[["col1","col2"]]
# new_df = selected_columns.copy()
#
# sns.scatterplot(x='sepal_length', y='sepal_width', hue='class', data=iris)
# sns.scatterplot(x='MEANX', y='SD', data=temp)
#
# df0 = pd.read_json(jsonstr,orient="index")
# df0 = pd.read_json('input.json',orient="index")
# df0.drop(to_drop, inplace=True)
# df0 = df0.transpose()
# 
# df_org[Y==-1]  #List Row for Y = -1 
#
# df0 = df[:1]  #For dummy prediction of testing data
# df0json = df0.to_json()
# file1 = open("x_input.json","w")
# file1.writelines(df0json)
# file1.close() 
#
# df.groupby('C_RISTIC').count()    #pandas count categories
# df.groupby('Parameter.Group').count()
#
# cpare = pd.DataFrame(compare)    #np array to dataframe
# cpare.to_csv('compare.csv')
#
# df.rename(columns={"A": "a", "B": "c"})

