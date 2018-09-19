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

* **K-nearest neighbors algorithm**, test acc : 0.9933
* **Artificial Neural Network**, test acc :  0.9775
* **Support Vector Machine 1vs1 approach**, test acc : 0.9459
* **Linear discriminant analyzis**, test acc : 0.7104


## BCI based system
![bci_system](https://user-images.githubusercontent.com/21131348/45600845-af172d00-ba03-11e8-8d69-a19c1f0ad02f.png)

## Artificial Neural Network - 97,7%

### Learning curve
![zrzut ekranu 2018-09-19 o 19 12 05](https://user-images.githubusercontent.com/21131348/45769773-e0c90780-bc40-11e8-85d8-c1741e8532d7.png)


### Confusion Matrix 
![zrzut ekranu 2018-09-19 o 19 12 42](https://user-images.githubusercontent.com/21131348/45769849-13730000-bc41-11e8-91fb-ffcf4a8cf3ba.png)


### Classification report

![zrzut ekranu 2018-09-19 o 19 11 37](https://user-images.githubusercontent.com/21131348/45769908-42897180-bc41-11e8-8f56-a9a4893371e8.png)

## K-nearest neighbors - 99,3%

### Neighbors test

![zrzut ekranu 2018-09-19 o 19 11 26](https://user-images.githubusercontent.com/21131348/45769983-7ebcd200-bc41-11e8-92fd-98abd6de0a36.png)

### Classification report

![zrzut ekranu 2018-09-19 o 19 12 58](https://user-images.githubusercontent.com/21131348/45770038-95fbbf80-bc41-11e8-9ff2-4a4f74d65c17.png)
