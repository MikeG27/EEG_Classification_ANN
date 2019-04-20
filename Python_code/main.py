#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:22:56 2018

@author: michal
"""

import os
import pandas_profiling

import modules.preprocessing
import config
from config import MAIN_DIR, PLOTS_DIR, DATA_DIR, DATA_RAW_DIR, DATA_PREPROCESSED_DIR, DATA_REPORTS
from modules import myutils, plots
from modules import models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from tpot import TPOTClassifier


def perform_dir_operations():
    os.chdir(config.MAIN_DIR)
    myutils.make_dir(PLOTS_DIR)
    myutils.make_dir(DATA_DIR)
    myutils.make_dir(DATA_RAW_DIR)
    myutils.make_dir(DATA_PREPROCESSED_DIR)
    myutils.make_dir(DATA_REPORTS)

    myutils.copy_folder("dataset.zip", DATA_RAW_DIR)
    myutils.unpack_zip("dataset.zip", DATA_RAW_DIR)
    os.remove(DATA_RAW_DIR + str("/dataset.zip"))


def get_class_treshold():
    class_treshold = myutils.open_csv(DATA_RAW_DIR + str("/no_signal.csv"))
    class_treshold = myutils.get_selected_feature(class_treshold)
    class_treshold = myutils.get_max_row_value(class_treshold)  # no signal limit(above signal exist)
    os.remove(DATA_RAW_DIR + str("/no_signal.csv"))
    os.chdir(MAIN_DIR)
    return class_treshold


def plot_subjects_EEG(preprocessedCSV):
    for id, subject in enumerate(preprocessedCSV.values()):
        plots.plot_feature_signal(subject, PLOTS_DIR, f"subject{id}", ylim=(2000, 8000))


def feature_selection(subjects):
    for id, subject in enumerate(subjects.values()):
        report = pandas_profiling.ProfileReport(subject)
        report.to_file(outputfile=f"{DATA_REPORTS}/subject{id + 1}.html")
        rejected_variables = report.get_rejected_variables(threshold=0.9)
        subject.drop(rejected_variables, axis=1, inplace=True)


def run_TPOT(subject):
    X_train, X_test, y_train, y_test = modules.preprocessing.process_to_training(subject, 0.3)
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_eeg_pipeline.py')


def Train_KNN(subject):
    X_train, X_test, y_train, y_test = modules.preprocessing.process_to_training(subject, 0.3)
    plots.test_neighbors(X_train, X_test, y_train, y_test) #dodać żeby zwracało optymalną liczbę sąsiadów dla tego algorytmu :)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train, y_train)
    KNN_pred = KNN.predict(X_test)
    evaluation_KNN = (accuracy_score(y_test, KNN_pred)) * 100
    print(f"KNN accuracy: {evaluation_KNN}")


def Train_ANN(subject):
    # Preprocess data
    X_train, X_test, y_train, y_test = modules.preprocessing.process_to_training(subject, test_set_size=0.3)
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


def run_clustering_kmeans(subject):
    # Do doczytania
    X_train, X_test, y_train, y_test = modules.preprocessing.process_to_training(subject, test_set_size=0.3)
    model = KMeans(n_clusters=4)

    # Fitting Model
    model.fit(X_train)

    # Prediction on the entire data
    y_pred = model.predict(X_test)

    evaluation_Kmeans = (accuracy_score(y_test, y_pred)) * 100
    print(f"Kmeans accuracy: {evaluation_Kmeans}")

    # Printing Predictions
    print(y_pred)
    print(y_test)


def run_preprocessing():
    perform_dir_operations()
    class_treshold = get_class_treshold()
    preprocessedCSV, class_dictionary = myutils.preprocess_subject_data(DATA_RAW_DIR, class_treshold)
    plot_subjects_EEG(preprocessedCSV)
    # plot_corelation_matrix
    feature_selection(preprocessedCSV)
    # preprocessedCSV = myutils.shuffle_data(preprocessedCSV)
    myutils.save_data(preprocessedCSV, DATA_PREPROCESSED_DIR)
    print("Data preprocessing was completed")

    # opcja do zapisywania plików w folderze raw


def run_classification():
    subject = "subject1"
    subject = myutils.open_csv(DATA_PREPROCESSED_DIR + "/" + str(subject) + ".csv")
    # run_TPOT(subject)
    run_clustering_kmeans(subject)
    # Train_KNN(subject)
    # TRAIN_ANN(subject)


if __name__ == "__main__":
    run_preprocessing()
    # run_classification()
