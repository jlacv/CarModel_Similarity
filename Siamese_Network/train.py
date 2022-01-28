# importing the necessary packages
import numpy as np
import cv2
import os
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from utils import config
from utils import net
from utils import utils
from utils.net import L1Dist #Created L1 dist layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Performing Data preprocessing
print("Data preprocessing...")

datatrain = []
labelstrain = []
#bboxestrain = []
imagePathstrain = []

datavalid = []
labelsvalid = []
#bboxesvalid = []
imagePathsvalid = []

datatest = []
labelstest = []
#bboxestest = []
imagePathstest = []


rows_train = open(config.ANNOTS_PATH + '/train_labels.csv').read().strip().split("\n")

# loop over the rows
for row in rows_train:
    # break the row into the filename, bounding box coordinates,
    # and class label
    row = row.split(",")
    (filename, startX, startY, endX, endY, label) = row

    # derive the path to the input image, load the image (in
    # OpenCV format), and grab its dimensions
    imagePath = os.path.sep.join([config.IMAGES_PATH,filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]


    # load the image and preprocess it
    image = load_img(imagePath,color_mode = "grayscale", target_size=(124,124))
    image = img_to_array(image)

    # update our list of data, class labels, bounding boxes, and
    # image paths
    datatrain.append(image)
    labelstrain.append(label)
    #bboxestrain.append((startX, startY, endX, endY))
    imagePathstrain.append(imagePath)

rows_valid = open(config.ANNOTS_PATH + '/valid_labels.csv').read().strip().split("\n")


for row in rows_valid:

    # break the row into the filename, bounding box coordinates,
    # and class label
    row = row.split(",")
    (filename1, startX1, startY1, endX1, endY1, label1) = row
    # derive the path to the input image, load the image (in
    # OpenCV format), and grab its dimensions
    imagePath1 = os.path.sep.join([config.IMAGES_PATH, filename1])
    image1 = cv2.imread(imagePath1)
    (h1, w1) = image1.shape[:2]

    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX1 = float(startX1) / w1
    startY1 = float(startY1) / h1
    endX1 = float(endX1) / w1
    endY1 = float(endY1) / h1

    # load the image and preprocess it
    image1 = load_img(imagePath1, color_mode = "grayscale", target_size=(124,124))
    image1 = img_to_array(image1)

    # update our list of data, class labels, bounding boxes, and
    # image paths
    datavalid.append(image1)
    labelsvalid.append(label1)
    #bboxesvalid.append((startX, startY, endX, endY))
    imagePathsvalid.append(imagePath1)


trainImages = np.array(datatrain, dtype="float32") / 255.0
trainLabels = np.array(list(map(int, labelstrain))) - 1 # Label encoding
#trainBBoxes = np.array(bboxestrain, dtype="float32")

validImages = np.array(datavalid, dtype="float32") / 255.0
validLabels = np.array(list(map(int, labelsvalid))) - 1
#validBBoxes = np.array(bboxesvalid, dtype="float32")

trainImages = np.expand_dims(trainImages, axis =-1)
validImages = np.expand_dims(validImages, axis =-1)
#testImages = np.expand_dims(testImages, axis =-1)

print("Preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainImages, trainLabels)
(pairValid, labelValid) = utils.make_pairs(validImages, validLabels)
#(pairTest, labelTest) = utils.make_pairs(testImages, testLabels)




# configuring the siamese network

print("Building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = net.make_embedding(config.IMG_SHAPE)

print ('The summary of the Featur extractor:', featureExtractor.summary())
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

l1 = L1Dist()
l1._name="distance_layer"
distance = l1(featsA, featsB)
classifier = Dense(1, activation="sigmoid")(distance)
siamese_net = Model(inputs=[imgA, imgB], outputs=classifier, name ='Siamese_Network')
print("Model Summary: ", siamese_net.summary())

print("Compiling model...")
siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=config.INIT_LR),
	metrics=["accuracy"])

print("Training model...")
history = siamese_net.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairValid[:, 0], pairValid[:, 1]], labelValid[:]),
	batch_size=config.BATCH_SIZE,
	epochs=config.EPOCHS)

print("Saving siamese model...")
siamese_net.save(config.MODEL_PATH)

# ploting the training history
print("Plotting training history...")
utils.plot_training(history, config.PLOTS_PATH)