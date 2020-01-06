# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 03:56:25 2018

@author: ahmed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

# fitting the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 100,random_state=0)
regressor.fit(X,Y)

# pridect with this model
y_pry= regressor.predict(6.5)

#visualaize the Random Forest regression
X_grid=np.arange(min(X),max(X),.1)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='black')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()