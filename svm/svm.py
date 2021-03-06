# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

# for loading the dataset as well as model
def input_dataset(name):
    data = pd.read_csv(name + '.csv')
    return data

# encoding columns
def lab_enc(x_train):
    y = 0
    num_list = []
    label = LabelEncoder()
    for(colName, colData) in x_train.iteritems():
        datatype = colData.dtype
        if(datatype == 'O'):
            if(len(set(colData)) < 20):
                x_train.iloc[:, y] = label.fit_transform(
                    x_train.iloc[:, y].astype(str))
                num_list.append(y)
            else:
                x_train = x_train.drop([colName], axis=1)
                y = y - 1
        y = y + 1
    return x_train, num_list

# one hot encoding of the columns that were encoded
def ohe_encode(x_train, y):
    if(len(y) == 0):
        return x_train
    else:
        ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
            categories='auto'), y)], remainder='passthrough')
        x_train = ct.fit_transform(x_train)
        return x_train
    
    
    
x = input_dataset('dataset') 
y = x.iloc[:, [10]] 
x = x.iloc[:, [1, 2, 3, 4, 8, 9]]  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)

x_train, numlist = lab_enc(x_train)
x_train = ohe_encode(x_train, numlist)

x_test, numlist = lab_enc(x_test)
x_test = ohe_encode(x_test, numlist)

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

result = []
ans = []

for i in y_pred:
    if (i == 'MANET'):
        result.append(0)
    elif (i == 'DTN'):
        result.append(1)
    else:
        result.append(2)
        
y_test = y_test.to_numpy()

for i in y_test:
    if (i == 'MANET'):
        ans.append(0)
    elif (i == 'DTN'):
        ans.append(1)
    else:
        ans.append(2)

print(accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(result, ans, pos_label=1)
print(metrics.auc(fpr, tpr))

