import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2 #pretrained deep learning model used by google
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

#MobileNetV2 requires 224×224 px input images.
IMG_SIZE = 224

#loading the pretrained model
def load_model():
    # Load pretrained MobileNetV2
    return MobileNetV2(weights="imagenet")
#ImageNet is a huge database of images used to train computer vision models.

#image processing
"""The preprocess function prepares the image so MobileNetV2 can read it.
It resizes the image to 224×224, turns it into a NumPy array, adds a batch dimension,
and applies the required preprocessing.
This makes sure the image is in the correct format for accurate predictions."""
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image(model, file):
    try:
        img = Image.open(file).convert("RGB")
    except:
        raise ValueError("Corrupted or unreadable image")

#decode_predictions picks the best answer and turns it into clear text.
#class_name = the predicted object (like “tabby” or “german shepherd”).
#confidence = how sure the model is about the prediction.
    x = preprocess(img)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=1)[0][0]

    class_name = decoded[1].lower()
    confidence = float(decoded[2])

    dog_keywords = [
        "dog", "retriever", "pug", "husky", "shepherd", "terrier",
        "chihuahua", "maltese", "bulldog", "dalmatian", "great_dane",
        "rottweiler", "doberman", "shih-tzu", "collie", "beagle",
        "boxer", "whippet"
    ]
    cat_keywords = ["cat", "tabby", "tiger_cat", "siamese", "persian", "lynx"]




    #decision for the prediction
    if any(k in class_name for k in cat_keywords):
        return "cat", confidence
    if any(k in class_name for k in dog_keywords):
        return "dog", confidence

    return "unknown", confidence


"""The GitHub model already gives “cat” or “dog”, so it doesn’t need this part.
MobileNetV2 gives breed names instead, so I added this code to turn those into “cat” or “dog”."""
