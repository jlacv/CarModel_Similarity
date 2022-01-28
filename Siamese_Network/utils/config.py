# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files

# Siamese Net
TRESHOLD = 0.7
IMG_SHAPE = (124, 124, 1)
EPOCHS = 3
BASE_OUTPUT ="Multitask_Model/output"

PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])


BASE_PATH = os.getcwd()
IMAGES_PATH = os.path.sep.join([BASE_PATH, "car_ims"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# define the path to the base output directory

# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "similarity.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 0.0001
BATCH_SIZE = 256

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
