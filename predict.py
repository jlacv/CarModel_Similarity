# USAGE
# !python /content/Multitask_Model/predict.py -f /content/car_ims/000137.jpg -s /content/car_ims/000020.jpg

# import the necessary packages
from Siamese_Network.utils.config import TRESHOLD
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from Siamese_Network.utils.net import L1Dist #Created L1 dist layer

import numpy as np
import mimetypes
import argparse
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--input1", required=True,
	help="full path to the first input images")
ap.add_argument("-s", "--input2", required=True,
	help="full path to the second input images")

args = vars(ap.parse_args())
#assume that we're working with
imagePaths = [args["input1"],args["input2"]]

# load our trained bounding box regressor from disk
model = load_model(os.getcwd() + "/Multitask_Model/output/Bboxregressor.h5")

image1 = load_img(imagePaths[0], target_size=(224, 224))
image1 = img_to_array(image1) / 255.0
image1 = np.expand_dims(image1, axis=0)

image2 = load_img(imagePaths[1], target_size=(224, 224))
image2 = img_to_array(image2) / 255.0
image2 = np.expand_dims(image2, axis=0)

# make bounding box predictions on the input image
preds1 = model.predict(image1)[0]
(startX1, startY1, endX1, endY1) = preds1

image1 = cv2.imread(imagePaths[0]) #image = imutils.resize(image, width=600)
(h1, w1) = image1.shape[:2]
startX1 = int(startX1 * w1)
startY1 = int(startY1 * h1)
endX1 = int(endX1 * w1)
endY1 = int(endY1 * h1)

# make bounding box predictions on the input image
preds2 = model.predict(image2)[0]
(startX2, startY2, endX2, endY2) = preds2

image1 = cv2.imread(imagePaths[1]) #image = imutils.resize(image, width=600)
(h2, w2) = image1.shape[:2]
startX2 = int(startX2 * w2)
startY2 = int(startY2 * h2)
endX2 = int(endX2 * w2)
endY2 = int(endY2 * h2)


#cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
print("The boundinx box prediction of the firt image is: ", (startX1, startY1, endX1, endY1))

print("The boundinx box prediction of the second image is: ", (startX2, startY2, endX2, endY2))


similarity_model = model = load_model(os.getcwd() + "/Multitask_Model/output/similarity.h5"
,custom_objects= {'L1Dist': L1Dist})
imageA = load_img(imagePaths[0],color_mode= "grayscale", target_size=(124, 124))
imageB = load_img(imagePaths[1], color_mode = "grayscale", target_size=(124, 124))



imageA = np.expand_dims(imageA, axis=0)
imageB = np.expand_dims(imageB, axis=0)
imageA = imageA / 255.0
imageB = imageB / 255.0

preds = model.predict([imageA, imageB])
proba = preds[0][0]

if proba < TRESHOLD:

   print("the two vehicles are not the same model")

else :
  print("the two vehicles are the same model")
