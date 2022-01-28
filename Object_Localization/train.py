from utils import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images

datatrain = []
bboxestrain = []
imagePathstrain = []

datavalid = []
bboxesvalid = []
imagePathsvalid = []

datatest = []
bboxestest = []
imagePathstest = []

# load the contents of the CSV annotations file
rows_train = open(config.ANNOTS_PATH + '/train_labels.csv').read().strip().split("\n")

# loop over the rows
for row in rows_train:
    # break the row into the filename, bounding box coordinates,
    # and class label
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row[0:5]

    # derive the path to the input image, load the image (in
    # OpenCV format), and grab its dimensions
    imagePath = os.path.sep.join([config.IMAGES_PATH,filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h

    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # update our list of data, class labels, bounding boxes, and
    # image paths
    datatrain.append(image)
    bboxestrain.append((startX, startY, endX, endY))
    imagePathstrain.append(imagePath)

rows_valid = open(config.ANNOTS_PATH + '/valid_labels.csv').read().strip().split("\n")

for row in rows_valid:

    # break the row into the filename, bounding box coordinates,
    # and class label
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row[0:5]
    # derive the path to the input image, load the image (in
    # OpenCV format), and grab its dimensions
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h

    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # update our list of data, class labels, bounding boxes, and
    # image paths
    datavalid.append(image)
    bboxesvalid.append((startX, startY, endX, endY))
    imagePathsvalid.append(imagePath)

rows_test = open(config.ANNOTS_PATH + '/test_labels.csv').read().strip().split("\n")



trainImages = np.array(datatrain, dtype="float32") / 255.0
trainBBoxes = np.array(bboxestrain, dtype="float32")

validImages = np.array(datavalid, dtype="float32") / 255.0
validBBoxes = np.array(bboxesvalid, dtype="float32")



'''
print("Saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(imagePathstest))
f.close()'''

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(flatten)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# train the network for bounding box regression
print("Training bounding box regressor...")
H = model.fit(
	trainImages[0:7001], trainBBoxes[0:7001],
	validation_data=(validImages[0:801], validBBoxes[0:801]),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("Saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# plot the model training history
N = config.NUM_EPOCHS
plt.style.use('dark_background')
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOTS_PATH)