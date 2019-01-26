# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Part1 Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part2 Make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialise ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform',input_dim = 11))
classifier.add(Dropout(rate=0.1))

# Second hidden layer
classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))

# Adding output layer
classifier.add(Dense(units = 1,activation= 'sigmoid',kernel_initializer='uniform'))

#Compile
classifier.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the dataset
classifier.fit(x=X_train,y=y_train,batch_size=10,epochs=100)

# Part 3 Making the prediction

# Predicting the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Example of Single prediction
# =============================================================================
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
# =============================================================================

new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)


# Part 4 Evaluating, Improving and Tuning the ANN

# Evaluating 
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier() :
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform',input_dim = 11))
    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform'))
    classifier.add(Dense(units = 1,activation= 'sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10 , epochs = 100)
# Train 10 ANN at the same time(parallrel)
accuracies = cross_val_score(estimator = classifier, X=X_train , y=y_train,cv = 10 , n_jobs=-1)
    
mean = accuracies.mean()
variance = accuracies.std()

#Tuning
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer) :
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform',input_dim = 11))
    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform'))
    classifier.add(Dense(units = 1,activation= 'sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss ='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size' : [25,32],'epochs' : [100,500] , 'optimizer' : ['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters , scoring='accuracy', cv=10)
grid_search = grid_search.fit(X=X_train,y=y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_




