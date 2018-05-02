import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import ceil
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
path = 'train.csv'
df = pd.read_csv(path)
df = df.drop('Id', axis = 1)

cols_to_transform = ['City','City Group','Type']
 
df[['Month','Day']] = df['Open Date'].str.split('/',n=1,expand = True)
df[['Day','Year']] = df['Day'].str.split('/',n=1,expand = True)
df =  df.drop(['Open Date'],axis = 1)

df_dummy = pd.get_dummies(df, columns = cols_to_transform)                 #converting the textual data to numeric one hot encoding
df_dummy = df_dummy.sample(frac=1)                                         #shuffling the rows
ylabel = df_dummy['revenue']
df_dummy = df_dummy.drop(columns=['revenue'])                              #removing the 'y' column from the original data

testData = pd.read_csv('test.csv')

trainingData = pd.read_csv('train.csv') 

testData_copy = testData

cols = [1]

trainingData.drop(trainingData.columns[cols],axis=1,inplace=True)            #dropping the id value column 
   
testData = testData.drop('Id', axis = 1)

#preprocessing steps
cols_to_transform = ['City','City Group','Type']                   #columns with texual information

testData[['Month','Day']] = testData['Open Date'].str.split('/',n=1,expand = True)
testData[['Day','Year']] = testData['Day'].str.split('/',n=1,expand = True)
testData =  testData.drop(['Open Date'],axis = 1)

testData = pd.get_dummies(testData, columns = cols_to_transform)            #encoding categorical values



missing_cols = set( testData.columns ) - set( df_dummy.columns )        #making sure no of columns in test and train are same
for c in missing_cols:
    df_dummy[c] = 0

missing_cols = set( df_dummy.columns ) - set( testData.columns )        #making sure no of columns in test and train are same
for c in missing_cols:
    testData[c] = 0




data_X = df_dummy.as_matrix(columns=None)                               #converting to numpy arrays
data_Y = ylabel.as_matrix(columns=None)

data_X = preprocessing.scale(data_X)  

train_X = data_X                                                        #dividing the data                                 
train_Y = data_Y



testData = testData.as_matrix(columns=None) 

testData = preprocessing.scale(testData)

#predicted_values = KRR.predict(testData)

clf = RandomForestRegressor(n_estimators=150)
clf.fit(train_X, train_Y)
pred = clf.predict(testData)
#pred = np.exp(pred)


predicted_df = pd.DataFrame({'Prediction':pred })  
id_column = testData_copy[['Id']]

out_df = pd.concat([id_column, predicted_df], axis=1)
out_df.to_csv('RF_out.csv',index=False)


