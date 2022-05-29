# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:00:06 2022

@author: ASUS
"""
from pandas import read_excel
dataset = read_excel('D:\\datanet.xlsx',header=0)
from sklearn.ensemble import RandomForestClassifier
fa = RandomForestClassifier(n_estimators=20)
x=dataset.iloc[:,0:96]
y=dataset.iloc[:,96]
from sklearn.model_selection import train_test_split
y_train, y_test, x_train, x_test= train_test_split(y,x,test_size=0.20)
arbre = fa.fit(x, y)
y_pred = arbre.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)