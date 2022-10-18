# EmbeddedAI
Projet IA Embarquée - Ayoub BEL HOUARY et Elie DAHER

# Introduction
Notre projet consiste à faire un aysteme d'intelligence artificielle quui va detecter le niveau de liquide (chloride de sodium) dans les boteilles pour la surveillance du niveau de remplissage.
Apres, il faut mettre ce systeme sur notre carte STM Discovery (STM32L4R9) et tester les inferences des images sur la carte elle meme.

# Datasets
Pour les donnees, on a telechargé une base de donnee fournie par ST qui contient des photos des bouteilles prisent de differents angles et avec des niveaux de liquides differents.
Ces donnees sont apres divisees en train et test sets avec lesquelles on va entrainer notre modele.
Pour notre modele, il etait conseiller de transformer les images en negative pour avoir de meilleurs detections de niveau.
Apres, on a fait la data augmentation des donnes pour generer plusieurs variations de chaque image et mieux generaliser notre modele.

