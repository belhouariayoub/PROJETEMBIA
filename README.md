---
# Projet IA Embarqué
   ### Ayoub BELHOUARI & Elie DAHER
   #### 18-10-2022
---

# Introduction
Notre projet consiste à faire un aysteme d'intelligence artificielle quui va detecter le niveau de liquide (chloride de sodium) dans les boteilles pour la surveillance du niveau de remplissage.
Apres, il faut mettre ce systeme sur notre carte STM Discovery (STM32L4R9) et tester les inferences des images sur la carte elle meme.

# Datasets
Pour les donnees, on a telechargé une base de donnee fournie par ST qui contient des photos des bouteilles prisent de differents angles et avec des niveaux de liquides differents reparties dans 4 dossiers differents (‘sal_data_100’, ‘sal_data_50’, ‘sal_data_80’, ‘sal_data_empty’).
Ces images sont apres transformees en numpy array de forme (4217, 64, 64, 3).
Ces donnees sont apres divisees en train et test sets avec lesquelles on va entrainer notre modele.
Pour notre modele, il etait conseiller de transformer les images en negative pour avoir de meilleurs detections de niveau.
Apres, on a fait la data augmentation des donnes pour generer plusieurs variations de chaque image et mieux generaliser notre modele.

# Modele

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
Apres qy’on a tester le model, on a eu une accuracy de 94.78 % et pas d’overfitting.
Alors, on a sauvegarder notre modele en “model.h5” ainsi que nos images et labels de tes (x_test.npy et y_test.npy) pour pouvoir les tester sur la STM32.
