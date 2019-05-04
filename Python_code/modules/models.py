#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:38:08 2018

@author: michal
"""


from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
import numpy as np 
import os
# =================================================================================================
#                                               INIT
# =================================================================================================

from keras.callbacks import Callback, EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose = 0, mode='auto')]


def get_model():
    
    # Adding the input layer and the first hidden layer
    model = Sequential()
    
    model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
     
    return model

def evaluate(model,X_test,y_test):
    
    dictionary = {}
    loss, accuracy = model.evaluate(X_test,y_test)
    
    dictionary["acc"] = accuracy
    dictionary["loss"] = loss

    return dictionary

def save_model(model,path,name):
    
    directory = os.path.join(path,"saved_models")
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save(directory + str("/") + name)
    
    return "Model was saved"
       
 
def predict(model,X_test,batch_size):
    y_pred = model.predict(X_test,batch_size)
    return y_pred



