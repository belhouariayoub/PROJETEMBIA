— 
# Projet IA Embarqué
## Ayoub BELHOUARI & Elie DAHER
## 18-10-2022
—

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
Epoch 4/50
99/99 [==============================] - 23s 233ms/step - loss: 1.1732 - accuracy: 0.4949 - val_loss: 1.0270 - val_accuracy: 0.5810
Epoch 5/50
99/99 [==============================] - 23s 232ms/step - loss: 1.0158 - accuracy: 0.5756 - val_loss: 0.8913 - val_accuracy: 0.6190
Epoch 6/50
99/99 [==============================] - 23s 233ms/step - loss: 0.8970 - accuracy: 0.6303 - val_loss: 0.7976 - val_accuracy: 0.6863
Epoch 7/50
99/99 [==============================] - 23s 232ms/step - loss: 0.7657 - accuracy: 0.6875 - val_loss: 0.6175 - val_accuracy: 0.7744
Epoch 8/50
99/99 [==============================] - 23s 233ms/step - loss: 0.6385 - accuracy: 0.7552 - val_loss: 0.5442 - val_accuracy: 0.7810
Epoch 9/50
99/99 [==============================] - 25s 249ms/step - loss: 0.5614 - accuracy: 0.7843 - val_loss: 0.4715 - val_accuracy: 0.8322
Epoch 10/50
99/99 [==============================] - 23s 234ms/step - loss: 0.4936 - accuracy: 0.8134 - val_loss: 0.4327 - val_accuracy: 0.8427
Epoch 11/50
99/99 [==============================] - 23s 232ms/step - loss: 0.4344 - accuracy: 0.8336 - val_loss: 0.4438 - val_accuracy: 0.8218
Epoch 12/50
99/99 [==============================] - 23s 233ms/step - loss: 0.3866 - accuracy: 0.8529 - val_loss: 0.3919 - val_accuracy: 0.8635
Epoch 13/50
99/99 [==============================] - 23s 234ms/step - loss: 0.3299 - accuracy: 0.8792 - val_loss: 0.3640 - val_accuracy: 0.8626
Epoch 14/50
99/99 [==============================] - 23s 234ms/step - loss: 0.2929 - accuracy: 0.8947 - val_loss: 0.3383 - val_accuracy: 0.8844
Epoch 15/50
99/99 [==============================] - 23s 234ms/step - loss: 0.2810 - accuracy: 0.8994 - val_loss: 0.2975 - val_accuracy: 0.8929
Epoch 16/50
99/99 [==============================] - 23s 232ms/step - loss: 0.2385 - accuracy: 0.9165 - val_loss: 0.2946 - val_accuracy: 0.8967
Epoch 17/50
99/99 [==============================] - 25s 254ms/step - loss: 0.2131 - accuracy: 0.9241 - val_loss: 0.3085 - val_accuracy: 0.8948
Epoch 18/50
99/99 [==============================] - 23s 234ms/step - loss: 0.1918 - accuracy: 0.9307 - val_loss: 0.2643 - val_accuracy: 0.8976
Epoch 19/50
99/99 [==============================] - 23s 234ms/step - loss: 0.1747 - accuracy: 0.9393 - val_loss: 0.2902 - val_accuracy: 0.8976
Epoch 20/50
99/99 [==============================] - 23s 233ms/step - loss: 0.1584 - accuracy: 0.9450 - val_loss: 0.2499 - val_accuracy: 0.9090
Epoch 21/50
99/99 [==============================] - 23s 233ms/step - loss: 0.1875 - accuracy: 0.9367 - val_loss: 0.2333 - val_accuracy: 0.9213
Epoch 22/50
99/99 [==============================] - 23s 235ms/step - loss: 0.1445 - accuracy: 0.9475 - val_loss: 0.2466 - val_accuracy: 0.9242
Epoch 23/50
99/99 [==============================] - 23s 233ms/step - loss: 0.1521 - accuracy: 0.9434 - val_loss: 0.2233 - val_accuracy: 0.9242
Epoch 24/50
99/99 [==============================] - 23s 235ms/step - loss: 0.1235 - accuracy: 0.9564 - val_loss: 0.2288 - val_accuracy: 0.9289
Epoch 25/50
99/99 [==============================] - 25s 254ms/step - loss: 0.1136 - accuracy: 0.9567 - val_loss: 0.2890 - val_accuracy: 0.9175
Epoch 26/50
99/99 [==============================] - 23s 236ms/step - loss: 0.0990 - accuracy: 0.9649 - val_loss: 0.2519 - val_accuracy: 0.9299
Epoch 27/50
99/99 [==============================] - 23s 235ms/step - loss: 0.1056 - accuracy: 0.9620 - val_loss: 0.2395 - val_accuracy: 0.9336
Epoch 28/50
99/99 [==============================] - 23s 234ms/step - loss: 0.1162 - accuracy: 0.9617 - val_loss: 0.2635 - val_accuracy: 0.9137
Epoch 29/50
99/99 [==============================] - 23s 234ms/step - loss: 0.1138 - accuracy: 0.9617 - val_loss: 0.2682 - val_accuracy: 0.9242
Epoch 30/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0948 - accuracy: 0.9677 - val_loss: 0.2370 - val_accuracy: 0.9270
Epoch 31/50
99/99 [==============================] - 23s 234ms/step - loss: 0.1013 - accuracy: 0.9665 - val_loss: 0.2492 - val_accuracy: 0.9299
Epoch 32/50
99/99 [==============================] - 25s 254ms/step - loss: 0.0826 - accuracy: 0.9728 - val_loss: 0.2269 - val_accuracy: 0.9346
Epoch 33/50
99/99 [==============================] - 24s 238ms/step - loss: 0.0801 - accuracy: 0.9760 - val_loss: 0.2210 - val_accuracy: 0.9346
Epoch 34/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0808 - accuracy: 0.9728 - val_loss: 0.2270 - val_accuracy: 0.9374
Epoch 35/50
99/99 [==============================] - 23s 236ms/step - loss: 0.0637 - accuracy: 0.9782 - val_loss: 0.2556 - val_accuracy: 0.9232
Epoch 36/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0569 - accuracy: 0.9794 - val_loss: 0.3139 - val_accuracy: 0.9100
Epoch 37/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0934 - accuracy: 0.9677 - val_loss: 0.2448 - val_accuracy: 0.9336
Epoch 38/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0693 - accuracy: 0.9738 - val_loss: 0.2204 - val_accuracy: 0.9431
Epoch 39/50
99/99 [==============================] - 24s 240ms/step - loss: 0.0617 - accuracy: 0.9763 - val_loss: 0.2722 - val_accuracy: 0.9308
Epoch 40/50
99/99 [==============================] - 25s 248ms/step - loss: 0.0641 - accuracy: 0.9760 - val_loss: 0.2456 - val_accuracy: 0.9327
Epoch 41/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0704 - accuracy: 0.9750 - val_loss: 0.3150 - val_accuracy: 0.9299
Epoch 42/50
99/99 [==============================] - 23s 233ms/step - loss: 0.0777 - accuracy: 0.9734 - val_loss: 0.2749 - val_accuracy: 0.9346
Epoch 43/50
99/99 [==============================] - 23s 237ms/step - loss: 0.0873 - accuracy: 0.9674 - val_loss: 0.2587 - val_accuracy: 0.9299
Epoch 44/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0602 - accuracy: 0.9801 - val_loss: 0.2944 - val_accuracy: 0.9299
Epoch 45/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0798 - accuracy: 0.9741 - val_loss: 0.2596 - val_accuracy: 0.9365
Epoch 46/50
99/99 [==============================] - 23s 233ms/step - loss: 0.0599 - accuracy: 0.9775 - val_loss: 0.2658 - val_accuracy: 0.9346
Epoch 47/50
99/99 [==============================] - 25s 254ms/step - loss: 0.0462 - accuracy: 0.9829 - val_loss: 0.2315 - val_accuracy: 0.9384
Epoch 48/50
99/99 [==============================] - 23s 235ms/step - loss: 0.0698 - accuracy: 0.9753 - val_loss: 0.2510 - val_accuracy: 0.9355
Epoch 49/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0602 - accuracy: 0.9801 - val_loss: 0.2776 - val_accuracy: 0.9355
Epoch 50/50
99/99 [==============================] - 23s 234ms/step - loss: 0.0473 - accuracy: 0.9836 - val_loss: 0.2151 - val_accuracy: 0.9479
