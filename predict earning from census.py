# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:52:45 2020

@author: ahmed
"""

"""
preprosessing  data 
for different ML model 

"""
import pandas as Pd

dataset =Pd.read_csv('census2.csv') # read the data set

target=dataset.iloc[:,-1].values
features=dataset.iloc[:,0:12].values
 
# transformation of data 
from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()

for x in list(range(1,8)):
    features[:,x]=labelEncoder.fit_transform(features[:,x])
    
    
features[:,11]=labelEncoder.fit_transform(features[:,11])
target=labelEncoder.fit_transform(target)

#spliting the  spliting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=.2,random_state=0)

# the classfier support vector machine
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)  
predict_data=model.predict(x_test)

# evaluated the model 
from sklearn.metrics import r2_score ,accuracy_score
print("the accuarcy is %3f"%accuracy_score(y_test,predict_data))
print("the R square is %3f"%r2_score(y_test,predict_data))

# classfier SVM with poly kernal

model2=SVC(kernel='poly',degree=4,C=0.01)
model2.fit(x_train,y_train)  
predict_data2=model2.predict(x_test)

print("the accuarcy is %3f"%accuracy_score(y_test,predict_data2))


# classfier SVM with  RBF kernel

model3=SVC(kernel='rbf',gamma=.5,C=0.01)
model3.fit(x_train,y_train)  
predict_data3=model3.predict(x_test)

print("the accuarcy is %3f"%accuracy_score(y_test,predict_data3))








    
    