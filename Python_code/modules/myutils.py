#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:53:50 2018

@author: michal
"""

import os
import zipfile
import shutil
import pandas as pd
from sklearn.utils import shuffle


def make_dir(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
def copy_folder(src,dst):
    shutil.copy(src,dst)

def unpack_zip(zip_name,extract_path):
    data_zip = zipfile.ZipFile(zip_name)
    data_zip.extractall(extract_path)

def open_csv(csv_name):
    data = pd.read_csv(csv_name)
    return data

def get_selected_feature(csv):
    csv = csv.iloc[: ,2:16]
    return csv

def get_max_row_value(data):
    return data.max().max()
    
def add_output_column(csv):
    
    """
    Create output column and add max value from each row
    
    """
    rowMax = (csv.max(axis=1)) # Max wartość z każdej linii
    csv["Output"] = rowMax
    return csv

def convert_to_categories(csv_output,n_class,no_signal):
    
    n_row = 0 #
    
    for output in csv_output:
        if output > no_signal:
            csv_output[n_row] = n_class
            #print(csv_output[n_row])
        else :
            csv_output[n_row] = 0
        
        n_row += 1
    
    return csv_output


def shuffle_data(subjects):
    for subject in subjects:
         subjects[subject] = shuffle(subjects[subject])
    return subjects


def save_data(subjects,folder):
    for subject_name, subject_eeg in subjects.items():
        csv_name = subject_name + ".csv"
        subject_eeg.to_csv(os.path.join(folder,csv_name))


def preprocess_subject_data(data_dir,no_signal_max):
    
    """
    Preprocess data from each subject,
    preprocess it and add to dictionary of subjects data
        
        * get only selected feature
        * get max_value from no_signal state
        * generate output columns with class numbers  
        
    Variables : 
        
        * data - dictionary contatins subject  data
        * subject_data - contain all of CSV files for each subject
        * feature - contain subject features like teeth or eyes_closed signal
        * n_class - number represents each class (ex 0 = no_signal, 1 = eyebrows ....)
        * subject_csv - list for preprocessed subject data
        * class dictionary - save number corresponded to class name
        
        
    """
    
    os.chdir(data_dir) # change directory to data_directory
    data = {} 
    class_dictionary = {}
    
    for subject in os.listdir(data_dir): # iterate over subjects 
        
        os.chdir(data_dir + str("/"+ subject)) # go to subject[i] folder
        subject_data = sorted(os.listdir(data_dir + str("/"+ subject))) # get list of sorted data of subject
        n_class = 0
        subject_csv = []
        
        for feature in subject_data: # petla ktora bierze poszczegolne dane
            
            n_class += 1
            csv = open_csv(feature) 
            csv = get_selected_feature(csv) # dane przyciete
            csv = add_output_column(csv)
            csv["Output"] = convert_to_categories(csv["Output"],n_class,no_signal_max)
            subject_csv.append(csv)        
            
            class_dictionary[feature] = n_class
            
        subject_csv = pd.concat(subject_csv,axis = 0)
        subject_csv = subject_csv[subject_csv.Output !=0]
        data[subject] = subject_csv 
    
    return data, class_dictionary


def preprocess_to_training(subject_data,test_set_size):
    
    X = subject_data.iloc[: ,0:14].values
    y = subject_data.iloc[: ,-1].values
        
    #Encoding categorical data 
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y).transform(y)

    
    #Splitting dataSet into the training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_set_size, random_state = 0)
    
    
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)

    return X_train,X_test,y_train,y_test



    
    