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

def get_important_features(csv):
    return csv.iloc[: ,2:16]

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


def save_data(subjects,folder):
    for subject_name, subject_eeg in subjects.items():
        csv_name = subject_name + ".csv"
        subject_eeg.to_csv(os.path.join(folder,csv_name))


def preprocess_subject_data(eeg_dir, signal_treshold):

    #TODO : Porozbijać to trochę :)

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
    eeg_signals = {}
    eeg_classes = {}
    eeg_subject_files = os.listdir(eeg_dir)

    for subject in eeg_subject_files: # iterate over subjects

        subject_eeg = sorted(os.listdir(os.path.join(eeg_dir,subject))) # get list of sorted data of subject
        subject_csv = []

        for eeg_id,eeg_class in enumerate(subject_eeg,start=1): # petla ktora bierze poszczegolne dane

            csv = open_csv(os.path.join(eeg_dir, subject, eeg_class))
            csv = get_important_features(csv)
            csv = add_output_column(csv)
            csv["Output"] = convert_to_categories(csv["Output"], eeg_id, signal_treshold)
            subject_csv.append(csv)

            eeg_classes[eeg_class] = eeg_id

        subject_csv = pd.concat(subject_csv,axis = 0)
        subject_csv = subject_csv[subject_csv.Output !=0]
        eeg_signals[subject] = subject_csv

    print("3.Subject data was preprocessed")
    return eeg_signals, eeg_classes





    
    