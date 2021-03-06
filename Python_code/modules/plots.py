#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:24:51 2018

@author: michal
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import os



def plot_feature_signal(feature_csv,save_fig,name,title_size = 15, figsize = (10,5),ylim = (3750,4750)):
    
    '''
    feature_csv = data of selected feature(teeth,eyesclosed...) in csv format
    savefig = path of file
    
    '''
    
    counter = 0 
    nameArray = list(feature_csv.columns.values)
    feature_csv = feature_csv.values
   
   
    fig = plt.figure(figsize = figsize)
    plt.xlabel("Samples")
    plt.ylabel('Voltage [μV]')
    plt.ylim(ylim)
    plt.grid()
    fig.suptitle(name + str(" EEG"),fontsize = title_size)
    
    while counter < len(nameArray) : 
        plt.plot(feature_csv[:,counter],label = nameArray[counter])
        plt.legend()
        counter = counter + 1

    if save_fig :
        plt.savefig(save_fig + str("/") + str(name))
            
    

def plot_training(history,save_fig,name,val = True,):
    
    """
    Comment
    """
    
    acc = history.history["acc"]
    loss = history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss = history.history["val_loss"]
    
    epochs = range(len(acc))
    
    plt.figure(2,figsize=(20,10))
    
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.legend()

    
    if save_fig:
         plt.savefig(save_fig + str("/") + str(name))

    plt.show()
        
  
        
def plot_confusion_matrix(model,X_test,y_pred,y_test,save_fig,name ):
    

    class_names = ["eyebrows","eyesClosed","smile","teeth"]
    
    y_pred = np.argmax(y_pred,axis = 1)
    y_pred = model.predict_classes(X_test)  
    y_test = np.argmax(y_test,axis = 1)
    cnf_matrix = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(20,12))
    plt.subplot(1,2,1)
    plt.title("Confusion Matrix")
    sns.heatmap(cnf_matrix,annot = True, xticklabels=class_names,
                      yticklabels=class_names,fmt=".1f",square=True,robust=True,cmap="Blues",
                      linewidths=4,linecolor='white')

    plt.subplot(1,2,2)
    cnf_matrix_normalized = cnf_matrix/cnf_matrix.sum(axis=0)
    plt.title("Confusion Matrix normalized")
    sns.heatmap(cnf_matrix_normalized,annot = True, xticklabels=class_names,
                      yticklabels=class_names,fmt="0f",square=True,robust=True,cmap="Blues",
                      linewidths=4,linecolor='white')
    
    if save_fig:
        plt.savefig(save_fig + str("/") + str(name))
        
    
def test_neighbors(X_train,X_test,y_train,y_test): 
    # TODO : Niech zwraca najlepsza liczbe sasiadow na tescie
    from sklearn.neighbors import KNeighborsClassifier
    
    training_accuracy = []
    test_accuracy = []
    
    neighbors_settings = range(1, 11)
      
    for n_neighbors in neighbors_settings:
        
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        KNN.fit(X_train, y_train)
        training_accuracy.append(KNN.score(X_train, y_train))
        test_accuracy.append(KNN.score(X_test, y_test))
    
    plt.figure(figsize=(15,5))
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()
    
    #print(type(test_accuracy))

        