# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:41:00 2021

@author: 41162395
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from sklearn.ensemble import IsolationForest
#from sklearn.preprocessing import MinMaxScaler

# Import raw dataset to pandas dataframe
df = pd.read_csv('ballshear_regression.csv')
# # Data cleasing drop NAN columns
# to_drop = ['LSL','USL','Parameter.Recipe','PROJECT_TYPE']
# df.drop(to_drop, inplace=True, axis=1)
# df.dropna(inplace=True)


#sns.set_style("dark")
# tips=sns.load_dataset('tips')
sns.jointplot(y='WIRE_SIZE', x='CUSTOMER',data=df,kind='reg')
sns.jointplot(y='WIRE_SIZE', x='MEANX',data=df,kind='reg')
sns.jointplot(y='WIRE_SIZE', x='SD',data=df,kind='reg')
sns.jointplot(y='MEANX', x='PRED',data=df,kind='kde')

# Scatterplot with continuous hues and sizes
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
g = sns.relplot(
    data=df,
    x="MEANX", y="PRED",
    hue="SD", size="WIRE_SIZE",
    palette=cmap, sizes=(10, 200),
)
#g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

# JOINT PLOT
sns.jointplot(
    data=df,
    x="MEANX", y="PRED", hue="WIRE_SIZE",
    kind="kde")
sns.jointplot(
    data=df,
    x="MEANX", y="PRED", hue="CUSTOMER",
    kind="kde")


cmp = pd.read_csv('compare.csv')
cmp_ideal = pd.read_csv('compare_ideal.csv')
sns.jointplot(y='0', x='y_pred',data=cmp,kind='kde')

sns.set_theme(style="whitegrid")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
               split=True, inner="quart", linewidth=1,
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)

sns.violinplot(data=df, x="WIRE_SIZE", y="MEANX", hue="CUSTOMER",
               split=True, inner="quart", linewidth=1,
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)

sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)

# Load the planets dataset and initialize the figure
planets = sns.load_dataset("planets")
g = sns.JointGrid(data=planets, x="year", y="distance", marginal_ticks=True)
g = sns.JointGrid(data=cmp, x="y", y="y_pred")


sns.displot(df, x="MEANX", y="PRED",hue="WIRE_SIZE")
sns.displot(df, x="MEANX", y="PRED",hue="WIRE_SIZE",kind="kde")


sns.pairplot(cmp)
sns.pairplot(cmp_ideal)

sns.displot(df, x="CUSTOMER", y="MEANX", hue="WIRE_SIZE", stat="probability")
sns.displot(df, x="CUSTOMER", y="PRED", hue="WIRE_SIZE", stat="probability")

sns.histplot(data=cmp, x="p_pred")
#############################
# pandas utility common command 
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