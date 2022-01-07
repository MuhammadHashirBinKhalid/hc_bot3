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
    myargparser.add_argument('--data_api', type=str, const='text', nargs='?', default='/robothon/mk7744/healthcare_roboplatform/bots/')
    myargparser.add_argument('--event_id', type=int, const=1, nargs='?', default=1)
    myargparser.add_argument('--result_path', type=str, const='text', nargs='?', 
                             default='D:/OneDrive - Higher Education Commission/Documents/NYU/Semester 3/ITP/Bots repos/models/bot')
    args = myargparser.parse_args()
    sys.path.append(args.data_api)
    bot_data = importlib.import_module('bot_data')
    print(args.maxtime)
    print(args.bot_id)
    print(args.data_api)
    mybotdata = bot_data.BotData(batch_size=30)
    result = mybotdata.fetchdataframe(1)
    result2 = mybotdata.fetchdataframe(2)
    result3 = mybotdata.fetchdataframe(3)
    result = result.append(result2, ignore_index=True)
    result = result.append(result3, ignore_index=True)
        
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    
    dataset=result.iloc[:,2:]
    dataset['hour']=dataset._time.dt.hour
    dataset['mins']=dataset._time.dt.minute
    X = dataset[['calories','distance','heart_rate','steps','hour','mins']].values
    y = dataset[['stress']].values
    
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
    filelocation = args.result_path
    #filelocation = 'models/bot'+str(args.bot_id)
    classifier.save(filelocation)
#    plt.show()
#    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#    cm = ConfusionMatrixDisplay.from_estimator(classifier,X_test,y_test)
    
    