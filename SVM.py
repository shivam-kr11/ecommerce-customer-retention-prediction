# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:53:58 2026

@author: HP
"""

#for high dimensional datasets 
#it consume more memory stacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

data=pd.read_csv(r"C:\Users\HP\Downloads\Social_Network_Ads.csv")

#Statistic/Descriptive
print(data.info())
print(data.describe())
print(data.head())
#Inferential
X=data[['Age','EstimatedSalary']]
y=data['Purchased']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
svm_lin=SVC(kernel='linear',random_state=42)
svm_lin.fit(X_train_scaled, y_train)
y_pred=svm_lin.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_pred)
print("Accuracy: ",accuracy_score(y_test,y_pred))

DecisionBoundaryDisplay.from_estimator(svm_lin,X_train_scaled,response_method='predict',
                                       cmap='coolwarm',alpha=0.5,X_label="Age(Scaled)",
                                       y_label="Estimated salary(scaled)")
plt.scatter(
    X_train_scaled[:,0],
    X_train_scaled[:,1],
    c=y_train,cmap='coolwarm',
    edgecolor='k',
    s=30)
plt.title("Linear SVM-Decision Boundry")
plt.tight_layout()
plt.show()



