# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 04:20:30 2020

@author: Isha
"""

# load dataset
import pandas as pd
import io
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score

df = pd.read_csv('dataset.csv')
print(df.shape)
# record columns to delete
to_del = ['Node', 'MAC', 'IP', 'network_id']
print(to_del)
# drop useless columns
df.drop(to_del, axis=1, inplace=True)
print(df.shape)
print(df)

#one-hot encoding
dataset_y = df['network_type']
dataset_x = df[['RAM', 'CPU', 'Battery', 'INTERNAL', 'network_class', 'RANGE']]

print(dataset_x)
print(dataset_y)
dataset_x = pd.concat([dataset_x,pd.get_dummies(dataset_x['network_class'], prefix='class')],axis=1)
dataset_x.drop(['network_class'],axis=1, inplace=True)
print(dataset_x)

#split the dataset
from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(dataset_x, dataset_y ,test_size = 0.2, shuffle=False)
X_train = X_train1.to_numpy()
X_test = X_test1.to_numpy()
Y_train = Y_train1.to_numpy()
Y_test = Y_test1.to_numpy()
print(X_train.shape)
print(Y_train.shape)

#training the model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)
print(Y_test1.shape)

#making a prediction
Y_pred = classifier.predict(X_test)
print(Y_test1.shape)
Y_test1["Predictions"] = Y_pred

#calculate accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

result = []
ans = []

for i in Y_pred:
    if (i == 'MANET'):
        result.append(0)
    elif (i == 'DTN'):
        result.append(1)
    else:
        result.append(2)
        
#Y_test = Y_test.to_numpy()

for i in Y_test:
    if (i == 'MANET'):
        ans.append(0)
    elif (i == 'DTN'):
        ans.append(1)
    else:
        ans.append(2)
print(result)
print(ans)

print(accuracy_score(Y_test, Y_pred))
fpr, tpr, thresholds = metrics.roc_curve(result, ans, pos_label=1)
print(metrics.auc(fpr, tpr))