# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:14:17 2021

Link :
    ANN regression model : https://colab.research.google.com/drive/1eje9zILprgVmohMN7cKykI3fn4FBRPnF
    seaborn : https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed
    Onehot encoder : https://towardsdatascience.com/columntransformer-in-scikit-for-labelencoding-and-onehotencoding-in-machine-learning-c6255952731b
                     https://stackoverflow.com/questions/58087238/valueerror-setting-an-array-element-with-a-sequence-when-using-onehotencoder-on
@author: 41162395
"""


# from numpy import array
# from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
#import seaborn as sns
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
to_drop = ['C_RISTIC','DATE_TIME','SHIFT','CUSTOMER','PT','EN_NO','DEVICE','REMARK','SD',
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
dfx = pd.read_json('x_input.json')  

# One hot encoder to numpy array X as below 9 columns or features
# Parameter.BondType,Parameter.Group,Parameter.No,Parameter.No_1,WIRE_SIZE,Parameter.Max,Parameter.Min,Parameter.Value, MEANX
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [0,1,2,3,4])], remainder='passthrough')
X = ct.fit_transform(df).toarray()  # upper case 'X'
X0 = ct.transform(dfx).toarray()  # upper case 'X0'

# Split X to 2 arrays as x inputFeatures and y outputResponse
selector = [i for i in range(X.shape[1]) if i != 35] #Column 35 is 'MEANX' 
x = X[:,selector]  # lower case 'x'  all column except 'MEANX'
y = X[:,35]  # lower case 'y'  for 'MEANX'

# Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Building the ANN
import tensorflow as tf
### Initializing the ANN
ann = tf.keras.models.Sequential()
### Adding the input layer and the first hidden layer"""
ann.add(tf.keras.layers.Dense(units=35, activation='relu'))
### Adding the second hidden layer"""
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
# ### Adding the third hidden layer"""
# ann.add(tf.keras.layers.Dense(units=35, activation='relu'))
# ### Adding the forth hidden layer"""
# ann.add(tf.keras.layers.Dense(units=35, activation='relu'))
# ### Adding the fifth hidden layer"""
# ann.add(tf.keras.layers.Dense(units=35, activation='relu'))
# # Adding the output layer"""
ann.add(tf.keras.layers.Dense(units=1))
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the ANN model on the Training set
ann.fit(X_train, y_train, batch_size = 8, epochs = 200)

# Predicting the results of the Test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

####################### Resume at here




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

