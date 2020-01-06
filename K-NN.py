# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:35:50 2018

@author: ahmed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# importing data set
dataset=pd.read_csv('Social_Network_Ads.csv')
trainData=dataset.iloc[:,2:4].values
targetData=dataset.iloc[:,4]
# preprocessing the data making the scaling for it 
from sklearn.preprocessing import StandardScaler 
scaler =StandardScaler()
scaler.fit_transform(trainData)

# spiliting the data  
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(trainData,targetData,test_size=.1,random_state=0)

# building the model 
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(x_train,y_train)

# make the prediction 
y_pred=classifier.predict(x_test)

# evaluate the model performance with the accuracy
from sklearn.metrics import accuracy_score ,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)


