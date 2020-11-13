# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:38:13 2020

@author: l
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('dataset.csv')

dataset.drop('MAC', axis='columns', inplace=True)
dataset.drop('IP', axis='columns', inplace=True)
dataset.drop('Node', axis='columns', inplace=True)
dataset.drop('network_id', axis='columns', inplace=True)

lb_make = LabelEncoder()
dataset["network_class"] = lb_make.fit_transform(dataset["network_class"])
dataset["network_type"] = lb_make.fit_transform(dataset["network_type"])
dataset["class"] = lb_make.fit_transform(dataset["class"])

X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, [6]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Feature Scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Applying classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Performance 
print(accuracy_score(y_test, y_pred))
