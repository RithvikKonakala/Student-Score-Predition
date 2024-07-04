# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:18:09 2024

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Classroom Data\June\25th June (Deployment - flask, joblib)\student_info.csv")

df =df.fillna(df.mean())

y = df.iloc[:,1:2]
X = df.iloc[:,0:1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_predict = regressor.predict(X_test)


pd.DataFrame(np.c_[X_test,y_test,y_predict],columns = ['study_hours','student_marks', 'predict_hours'])

import joblib
joblib.dump(regressor, "pedict_university.pkl")

import os
os.getcwd()
