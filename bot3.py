# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:22:45 2021

@author: moham
"""
import argparse
import datetime
import logging
import math
from time import sleep
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import importlib

if __name__ == "__main__":
    # market_event_securities = ["GEH0:MBO","GEM2:MBO","GEU0:MBO"]
    myargparser = argparse.ArgumentParser()
    myargparser.add_argument('--maxtime', type=int, const=120, nargs='?', default=120)
    myargparser.add_argument('--bot_id', type=str, const='text', nargs='?', default=3)
    myargparser.add_argument('--data_api', type=str, const='text', nargs='?', default='bot_data')
    args = myargparser.parse_args()
    
    bot_data = importlib.import_module(args.data_api)
    print(args.maxtime)
    print(args.bot_id)
    print(args.data_api)
    mybotdata = bot_data.BotData(batch_size=30)
    result = mybotdata.fetchdataframe()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    
    dataset=result.iloc[:,2:]
    dataset['hour']=dataset._time.dt.hour
    dataset['mins']=dataset._time.dt.minute
    dataset['stress'] = dataset.heart_rate>100
    print(dataset.head())
#    sys.exit()
    X = dataset.iloc[:,1:7].values
    y = dataset.iloc[:,7].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    
    # Adding the second hidden layer
    #classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
    
    # Part 3 - Making the predictions and evaluating the model
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    filelocation = 'models/bot'+str(args.bot_id)
    classifier.save(filelocation)
#    plt.show()
#    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#    cm = ConfusionMatrixDisplay.from_estimator(classifier,X_test,y_test)
    
    