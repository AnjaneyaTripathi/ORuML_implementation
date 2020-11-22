import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import warnings

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
y_test = y_test.to_numpy()

"""
#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(x_train,y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

"""

classifier = KNeighborsClassifier(n_neighbors=1,leaf_size=1,p=1)
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
        
       

for i in y_test:
    if (i == 'MANET'):
        ans.append(0)
    elif (i == 'DTN'):
        ans.append(1)
    else:
        ans.append(2)

print("Accuracy:",accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(result, ans, pos_label=2)
print("AUC:",metrics.auc(fpr, tpr))


