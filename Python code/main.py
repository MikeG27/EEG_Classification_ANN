#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:22:56 2018

@author: michal
"""

import os
import pandas_profiling
from modules import myutils, plots
from modules import models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

MAIN_DIR = os.getcwd()
PLOTS_DIR = os.path.join(MAIN_DIR, "Plots")
DATA_DIR = os.path.join(MAIN_DIR, "dataset")


def perform_dir_operations():
    # TODO : Podzielic to bardziej
    os.chdir(MAIN_DIR)
    myutils.make_dir(PLOTS_DIR)
    myutils.make_dir(DATA_DIR)
    myutils.copy_folder("dataset.zip", DATA_DIR)
    myutils.unpack_zip("dataset.zip", DATA_DIR)
    os.remove(DATA_DIR + str("/dataset.zip"))
    # NO SIGNAL DATA
    no_signal = myutils.open_csv(DATA_DIR + str("/no_signal.csv"))
    no_signal = myutils.get_selected_feature(no_signal)
    # plots.plot_feature_signal(no_signal,plots_path,"no_signal")
    no_signal_max = myutils.get_max_row_value(no_signal)  # no signal limit(above signal exist)
    os.remove(DATA_DIR + str("/no_signal.csv"))

    os.chdir(MAIN_DIR)

    return no_signal


def plot_subjects_EEG(preprocessedCSV):
    plots.plot_feature_signal(preprocessedCSV["subject1"], PLOTS_DIR, "subject1", ylim=(2000, 8000))
    plots.plot_feature_signal(preprocessedCSV["subject2"], PLOTS_DIR, "subject2", ylim=(2000, 8000))
    plots.plot_feature_signal(preprocessedCSV["subject3"], PLOTS_DIR, "subject3", ylim=(2000, 8000))
    plots.plot_feature_signal(preprocessedCSV["subject4"], PLOTS_DIR, "subject4", ylim=(2000, 8000))
    plots.plot_feature_signal(preprocessedCSV["subject5"], PLOTS_DIR, "subject5", ylim=(2000, 8000))


def feature_selection(subject):
    # Compute pandas profiling raport
    profile = pandas_profiling.ProfileReport(subject)
    profile.to_file(outputfile="subject1.html")
    # Drop rejected variables
    rejected_variables = profile.get_rejected_variables(threshold=0.9)
    subject_1_after_f_selection = subject.drop(rejected_variables, axis=1)

    return subject_1_after_f_selection



def Train_KNN(subject):
    X_train, X_test, y_train, y_test = myutils.preprocess_to_training(subject, 0.3)
    plots.test_neighbors(X_train, X_test, y_train, y_test)  #
    # the best numer of neigbours is 1 so..
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train, y_train)
    KNN_pred = KNN.predict(X_test)
    evaluation_KNN = (accuracy_score(y_test, KNN_pred)) * 100
    print(f"ANN accuracy: {evaluation_KNN}")



def TRAIN_ANN(subject):
    # Preprocess data
    X_train, X_test, y_train, y_test = myutils.preprocess_to_training(subject, test_set_size=0.3)
    # Get model and train
    model = models.get_model()
    model.summary()
    history = model.fit(X_train, y_train, batch_size=128, epochs=200, validation_split=0.2)

    plots.plot_training(history, PLOTS_DIR, "training")
    models.save_model(model, MAIN_DIR, "model1.h5")
    evaluation_ANN = models.evaluate(model, X_test, y_test)
    print(f"\nANN accuracy: {evaluation_ANN}")
    # Plot confusion matrix
    y_pred = models.predict(model, X_test, 128)
    plots.plot_confusion_matrix(model, X_test, y_pred, y_test, PLOTS_DIR, "confusion matrix")


if __name__ == "__main__":
    no_signal_max = perform_dir_operations()
    preprocessedCSV, class_dictionary = myutils.preprocess_subject_data(DATA_DIR, no_signal_max)
    plot_subjects_EEG()
    subject = feature_selection(preprocessedCSV["subject1"])
    Train_KNN(subject)
    TRAIN_ANN(subject)
