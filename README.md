## French Version: 
Ce projet implémente une API Flask capable de prédire si une image représente un chat, un chien, ou unknown.
L’API utilise le modèle MobileNetV2 préentraîné sur ImageNet pour effectuer la classification des images.
Le fichier app.py gère l’API :il reçoit l’image envoyée par l’utilisateur,vérifie son extension et sa taille, la sauvegarde temporairement, et appelle la fonction de prédiction.
Le fichier model.py contient toute la logique liée au modèle :
chargement du modèle MobileNetV2, prétraitement de l’image (redimensionnement, normalisation), exécution de la prédiction, interprétation du résultat pour retourner cat, dog, ou unknown.
Le fichier image_utils.py vérifie que le fichier uploadé est bien une image valide (jpg, jpeg, png).
L’API renvoie toujours une réponse JSON structurée et gère les erreurs (fichier manquant, mauvaise extension, image corrompue, échec du modèle).

## English Version: 
This project implements a Flask API that predicts whether an uploaded image contains a cat, a dog, or unknown.
The API uses the MobileNetV2 model pretrained on ImageNet to perform image classification.
The app.py file handles the API logic: it receives the uploaded image, checks the file type and size, saves it temporarily, and calls the prediction function.
The model.py file contains the model logic:
loading the MobileNetV2 model, preprocessing the image (resizing, normalization), running inference, interpreting the output to return cat, dog, or unknown.
The image_utils.py file ensures that the uploaded file is a valid image (jpg, jpeg, png).
The API always returns a structured JSON response and includes error handling for missing files, invalid formats, corrupted images, or prediction failures.
