#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:22:56 2018

@author: michal
"""

import os 
import numpy as np
from modules import myutils, plots
from modules import models


# =============================================================================
#                           Folder operations
# =============================================================================

main_path = "/home/michal/Pulpit/Koncowe EEG" 
plots_path = "/home/michal/Pulpit/Koncowe EEG/Plots" 
os.chdir(main_path)

data_dir = os.path.join(main_path,"dataset")

myutils.make_dir(plots_path)
myutils.make_dir(data_dir)
myutils.copy_folder("dataset.zip",data_dir)
myutils.unpack_zip("dataset.zip",data_dir)
os.remove(data_dir + str("/dataset.zip"))

# NO SIGNAL DATA
no_signal = myutils.open_csv(data_dir + str("/no_signal.csv"))
no_signal = myutils.get_selected_feature(no_signal)
#plots.plot_feature_signal(no_signal,plots_path,"no_signal")
no_signal_max = myutils.get_max_row_value(no_signal) # no signal limit(above signal exist)
os.remove(data_dir + str("/no_signal.csv"))

preprocessedCSV, class_dictionary = myutils.preprocess_subject_data(data_dir,no_signal_max)
os.chdir(main_path)

'''
Plot and save all subjects signal
plots.plot_feature_signal(preprocessedCSV["subject1"],plots_path,"subject1",ylim = (2000,8000))
plots.plot_feature_signal(preprocessedCSV["subject2"],plots_path,"subject2",ylim = (2000,8000))
plots.plot_feature_signal(preprocessedCSV["subject3"],plots_path,"subject3",ylim = (2000,8000))
plots.plot_feature_signal(preprocessedCSV["subject4"],plots_path,"subject4",ylim = (2000,8000))
plots.plot_feature_signal(preprocessedCSV["subject5"],plots_path,"subject5",ylim = (2000,8000))
'''


# =============================================================================
#                            Model for 1 person
# =============================================================================


# Preprocess data
X_train,X_test,y_train,y_test = myutils.preprocess_to_training(preprocessedCSV["subject1"],test_set_size = 0.3)

#Get model and train
model = models.get_model()
model.summary()
history = model.fit(X_train, y_train,batch_size = 128 , epochs = 100,validation_split = 0.3)
#model.save()
plots.plot_training(history,plots_path,"training")
models.save_model(model,main_path,"model1")
evaluation = models.evaluate(model,X_test,y_test)

#Plot confusion matrix 
y_pred = models.predict(model,X_test,128) 
plots.plot_confusion_matrix(model,X_test,y_pred,y_test,plots_path,"confusion matrix")






