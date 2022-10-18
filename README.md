---
# Projet IA Embarqué
   ### Ayoub BELHOUARI & Elie DAHER
   #### 18-10-2022
---

# Introduction

Ce projet décrit l'implémentation d'un modèle de réseaux neurone basés sur la base de donnée Salinebottle sur UNE carte STM Discovery (STM32L4R9). Il contient l'archive du projet et les scripts python pour construire le modèle et communiquer avec la carte. 
L'objectif c'est de detecter le niveau de liquide (chloride de sodium) dans les boteilles pour la surveillance du niveau de remplissage avec une IA embarquée.

# Datasets

La dataset a été fournit par ST,elle contient des photos des bouteilles prisent de differents angles et avec des niveaux de liquides differents reparties dans 4 dossiers differents (‘sal_data_100’, ‘sal_data_50’, ‘sal_data_80’, ‘sal_data_empty’).
Les données d'image, pour chaque niveau de remplissage de la bouteille, fournissent différentes perspectives, conditions d'éclairage, mise au point sur la bouteille, arrière-plan. Ces éléments sont utiles pour vérifier l'évidence visuelle du niveau de liquide salin à l'intérieur de la bouteille.
L'ensemble de données proposé consiste en une archive de 4217 images.

![graph-accuracy vs epoch for test and validation](img/contents_of_data.jpeg#center)

Les images de la dataset seront transformée en vecteur numpy array de taille (4217, 64, 64, 3).
Ces donnees sont apres divisees en train et test sets avec lesquelles on va entrainer notre modele.
Pour notre modele, il etait conseiller de transformer les images en negative pour avoir de meilleurs detections de niveau.
Apres, on a fait la data augmentation des donnes pour generer plusieurs variations de chaque image et mieux generaliser notre modele.
 
# Modele
## Modele V1

```
Model: "sequential_38"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_64 (Conv2D)          (None, 64, 64, 32)        896       
                                                                 
 conv2d_65 (Conv2D)          (None, 62, 62, 32)        9248      
                                                                 
 max_pooling2d_32 (MaxPoolin  (None, 31, 31, 32)       0         
 g2D)                                                            
                                                                 
 dropout_46 (Dropout)        (None, 31, 31, 32)        0         
                                                                 
 conv2d_66 (Conv2D)          (None, 31, 31, 64)        18496     
                                                                 
 conv2d_67 (Conv2D)          (None, 31, 31, 64)        36928     
                                                                 
 max_pooling2d_33 (MaxPoolin  (None, 15, 15, 64)       0         
 g2D)                                                            
                                                                 
 dropout_47 (Dropout)        (None, 15, 15, 64)        0         
                                                                 
 flatten_19 (Flatten)        (None, 14400)             0         
                                                                 
 dense_38 (Dense)            (None, 512)               7373312   
                                                                 
 dropout_48 (Dropout)        (None, 512)               0         
                                                                 
 dense_39 (Dense)            (None, 4)                 2052      
                                                                 
=================================================================
Total params: 7,440,932
Trainable params: 7,440,932
Non-trainable params: 0
```
Ce modele avait une accuracy de 88.06% et il avait un peut overfit comme le montre le graph de la figure ci-dessous:

![graph-accuracy vs epoch for test and validation](img/graph_mod1.jpg)

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 64, 64, 32)        896       
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 32, 32, 32)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 30, 30, 32)        9248      
                                                                 
 spatial_dropout2d_1 (Spatia  (None, 30, 30, 32)       0         
 lDropout2D)                                                     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 15, 15, 32)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 15, 15, 64)        18496     
                                                                 
 conv2d_7 (Conv2D)           (None, 15, 15, 64)        36928     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 7, 7, 64)          0         
                                                                 
 flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
 dense_2 (Dense)             (None, 128)               401536    
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 467,620
Trainable params: 467,620
Non-trainable params: 0
_________________________________________________________________
Epoch 1/50
99/99 [==============================] - 23s 228ms/step - loss: 1.3891 - accuracy: 0.2533 - val_loss: 1.3853 - val_accuracy: 0.2474
Epoch 2/50
99/99 [==============================] - 25s 253ms/step - loss: 1.3842 - accuracy: 0.2720 - val_loss: 1.3845 - val_accuracy: 0.2844
Epoch 3/50
99/99 [==============================] - 23s 234ms/step - loss: 1.3182 - accuracy: 0.3624 - val_loss: 1.3109 - val_accuracy: 0.4076
.
.
Epoch 49/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0602 - accuracy: 0.9801 - val_loss: 0.2776 - val_accuracy: 0.9355
Epoch 50/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0473 - accuracy: 0.9836 - val_loss: 0.2151 - val_accuracy: 0.9479
```

## Model Accuracy
Apres avoir tester le model, L'accuracy du modèle est  94.78 % et d'aprés le graphe on remarque que le modéle n'overfitt pas.

![graph-accuracy vs epoch for test and validation](img/graph_mod2.jpg)

Afin d'embarquer le modéle sur la carte STM32 nous avons sauvegarder le modele sous format h5 “model.h5” ainsi que les  images et labels pour le test (x_test.npy et y_test.npy).







# Exemple contradictoire utilisant FGSM : 

Afin de tester la sécurité et l'intégrité de notre modéle nous avons appliquée un exemple d’attaque contradictoire à l’aide de l’attaque Fast Gradient Signed Method (FGSM).
La méthode du signe de gradient rapide fonctionne en utilisant les gradients du réseau de neurones pour créer un exemple contradictoire.  




