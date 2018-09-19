## This repository contain solution for categorical classification problem of EGG biomedical artifacts 

In Jupyter notebook author presents the results of work related to the problem of classification biological artifacts using artificial neural networks. For this purpose author through Emotiv Epoc+ NeuroHeadset collected measurement that has been processed and given to the learning process. The detailed scope of work included defining the problem of work and the method of solving it, collecting measurement data, processing measurement data, creating the structure of an artificial neural network, selecting hyperparameters and performing tests and drawing the appropriate conclusions. The obtained results may contribute to further research or implementation of the neural network model in order to control based on facial expressions. 

![3](https://user-images.githubusercontent.com/21131348/44955960-623c3e00-aebc-11e8-8e18-ad11f80edc63.png)

## Problem Overview 

Problem definition : Implementation of the neural network model in order to control based on facial expressions. 

* **Problem type** : Multiclass clasiffication
* **Number of features** : 12 electrodes
* **Number of class** : 4 facial expressions
* **Class type**  : "eyes_closed, eyebrows, smile, teeth"
 
## Used models

* **K-nearest neighbors algorithm**, test acc : 0,9933
* **Artificial Neural Network**, test acc :  0.9775
* **Support Vector Machine 1vs1 approach**, test acc : 94.5923
* **Linear discriminant analyzis**, test acc : 71.04825


## BCI based system
![bci_system](https://user-images.githubusercontent.com/21131348/45600845-af172d00-ba03-11e8-8d69-a19c1f0ad02f.png)

## Results

### Learning curve
![zrzut ekranu 2018-09-2 o 15 12 57](https://user-images.githubusercontent.com/21131348/44956359-ef829100-aec2-11e8-87ac-5606471e19bd.png)

### Confusion Matrix 
![zrzut ekranu 2018-09-2 o 15 01 23](https://user-images.githubusercontent.com/21131348/44956245-340d2d00-aec1-11e8-8437-3c3b7fcdc2a8.png)


