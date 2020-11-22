import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn import preprocessing
from keras.models import model_from_json
from sklearn.metrics import accuracy_score

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

result = [] 

for i in y['network_type']:
    if (i == 'MANET'):
        result.append(0)
    elif (i == 'DTN'):
        result.append(1)
    else:
        result.append(2)
        
x, numlist = lab_enc(x)
x = ohe_encode(x, numlist)

x = x.astype('float32')
        
x = x.toarray() #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x = pd.DataFrame(x_scaled)

y = pd.DataFrame(result)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=4)
'''
model = Sequential()
model.add(Dense(24, activation='relu', input_dim=22, kernel_initializer='uniform'))
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
model.summary()
'''

sgd = SGD(lr = 0.007, momentum = 0.5)
'''
model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size = 1, epochs = 40, verbose=2)

scores = model.evaluate(x_train, y_train, batch_size=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(x_test, y_test, batch_size=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# list all data in history
#print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'loss'], loc='upper left')
plt.show()
'''

'''
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

'''
 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
loaded_model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
score = loaded_model.evaluate(x_test, y_test, batch_size=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

res = loaded_model.predict(x_test)
y_pred = []

for i in res:
    m = max(i)
    if(m == i[0]):
        y_pred.append(0)
    elif(m == i[1]):
        y_pred.append(1)
    else:
        y_pred.append(2)

print(accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_test, pos_label=2)
print(metrics.auc(fpr, tpr))















