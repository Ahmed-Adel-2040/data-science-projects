# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 02:07:23 2018

@author: ahmed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# importing data set
dataset=pd.read_csv('50_Startups.csv')
trainData=dataset.iloc[:,:-1].values
targetData=dataset.iloc[:,4].values

# this the class wihich make the dummy encoder
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelEncoder_train=LabelEncoder()
trainData[:,3]=labelEncoder_train.fit_transform(trainData[:,3])

# make the country data to be Dummy encoder
oneHotEncoder=OneHotEncoder(categorical_features=[3])
trainData=oneHotEncoder.fit_transform(trainData).toarray()

# avoiding the dummy variable trap 
trainData=trainData[:,1:]

#  spilting the data into train data and test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(trainData,targetData,test_size=.1,random_state=0)

# fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predict on the model
preidct_Data=regressor.predict(x_test)

#accuracy=regressor.score(y_test, preidct_Data, sample_weight=None)

# test the mean square error of the linear regression 
from sklearn.metrics import r2_score
print("the Accuracy of the model: %.2f"  % r2_score(y_test, preidct_Data))
     









