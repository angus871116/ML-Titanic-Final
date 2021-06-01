# -*- coding: utf-8 -*-
"""
Created on Fri May 28 22:33:15 2021

@author: angus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
plt.style.use( 'ggplot' ) 

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.isnull().sum())
print(df_test.isnull().sum())

#%% 前處理 fill missing value
def fill_age_by_title(data_df):
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
    # replace rare titles with more common ones
    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
              'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
    
    data_df.replace({'Title': mapping}, inplace=True)
    
    titles = list(data_df.Title.value_counts().index)

    # for each title, impute missing age by the median of the persons with the same title
    for title in titles:
        age_to_impute = data_df.groupby('Title')['Age'].median()[title]
        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
        
def encode(data_df):
    data_df['Sex'].replace(['male','female'],[0,1],inplace=True)
    data_df['Embarked'].fillna('S') #S人最多 先猜缺失值都填S
    data_df['Embarked'].replace(['S','C', 'Q'], [1, 2, 3],inplace=True)
    
def age_cut(data_df):
    data_df['AgeBin'] = pd.cut(data_df['Age'], 5) #把age切割
    data_df['AgeBin_Code'] = LabelEncoder().fit_transform(data_df['AgeBin'])
    
def drop_cabin(data_df):
    data_df.drop(['Cabin'], 1, inplace=True)
    
def fare(data_df):
    data_df['Fare'].fillna('7.5',inplace=True) #缺的那個人是3等艙 隨便猜一個7.5
    #還需要把套票除上重複數量 取得每一個人所付的實際價格
    
fill_age_by_title(df_train)
encode(df_train)
age_cut(df_train)
drop_cabin(df_train)
fare(df_train)
    
#%% 特徵分析

#連續特徵分布狀況
continuous_numeric_features = ['Age', 'Fare', 'Parch', 'SibSp']
for feature in continuous_numeric_features:
    sns.distplot(df_train[feature])
    plt.show()

#類別型特徵對生存率作圖    
selected_cols = ['Sex','Pclass','AgeBin_Code','Embarked','SibSp','Parch']

plt.figure( figsize=(10,len(selected_cols)*5) )
gs = gridspec.GridSpec(len(selected_cols),1)    
for i, col in enumerate( df_train[selected_cols] ) :        
    ax = plt.subplot( gs[i] )
    sns.countplot( df_train[col], hue=df_train.Survived, palette=['lightcoral','skyblue'] )
    ax.set_yticklabels([])
    ax.set_ylabel( 'Counts' )
    ax.legend( loc=1 )   # upper right:1 ; upper left:2
    for p in ax.patches:
        ax.annotate( '{:,}'.format(p.get_height()), (p.get_x(), p.get_height()+1.5) )
plt.show()


#特徵相關性分析Pearson correlation
features=['Age','Fare','Parch','Sex','SibSp','Embarked']    
target='Survived'
columns = df_train[features + [target]].columns.tolist()
nColumns = len(columns)
result = pd.DataFrame(np.zeros((nColumns, nColumns)), columns=columns)

for col_a in range(nColumns):
    for col_b in range(nColumns):
        result.iloc[[col_a], [col_b]] = pearsonr(df_train.loc[:, columns[col_a]], df_train.loc[:,  columns[col_b]])[0]
        
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(result, yticklabels=columns, vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=.2)
ax.set_title('PCC - Pearson correlation coefficient')
plt.show()

print(df_train.isnull().sum())

#%%        