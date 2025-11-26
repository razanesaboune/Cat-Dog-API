#image validation and preprocessing
#import image from pillow library to open, resize, and handle image files.
import os
from PIL import Image
import numpy as np

#Creates a set containing allowed file extensions.
#We will use this to check if the uploaded file is an image (JPEG or PNG).
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
#The allowed_file function checks whether the uploaded file is really an image.
#It makes sure the filename ends with .jpg, .jpeg, or .png.
#his stops users from uploading wrong or dangerous files like .exe, .txt, or .pdf.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# This function returns True if the file has a valid extension, otherwise False.




#This function prepares an image to be used by the model.
#Models cannot read raw image
#It resizes the image to 150×150, 
# converts it into a normalized array of numbers, 
# and adds an extra dimension so the model can process it correctly.
def preprocess_image(image_path):
    img = Image.open(image_path).resize((150, 150)) #wide, height
    img = np.array(img) / 255.0
    #The model expects a batch of images, not a single image.
    #expand_dims adds a dimension at the front:
    img = np.expand_dims(img, axis=0)
    return img
