# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
# Building the Distance Layer

class L1Dist(Layer):

  def __init__(self, **kwargs):
    super().__init__()
  def call(self, A_embedding, B_embedding):
    return abs(A_embedding - B_embedding)

def make_siamese_model():
  # Anchor image input in the network
  image1 = Input(name='img1', shape=config.IMG_SHAPE)

  # Validation image in the network
  image2 = Input(name='img2', shape=config.IMG_SHAPE)

  # Combine siamese distance components
  l1 = L1Dist()
  l1._name = 'l1_distance'
  distances = l1(make_embedding(image1), make_embedding(image2))

  # Classification layer
  classifier = Dense(1, activation='sigmoid')(distances)

  return Model(inputs=[image1, image2], outputs=classifier, name='Siamese_Net')

def make_embedding(inputShape ):

# specify the inputs for the feature extractor network
  inputs = Input(inputShape, name='input_image')


  c1 = Conv2D(64, (10,10), activation='relu')(inputs)
  m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
  c2 = Conv2D(128, (7,7), activation='relu')(m1)
  m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
  c3 = Conv2D(128, (4,4), activation='relu')(m2)
  m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

  #Final Embedding Block
  c4 = Conv2D(256, (4,4), activation='relu')(m3)
  f1 = Flatten()(c4)
  d1 = Dense(4096, activation='sigmoid')(f1)
  
  # prepare the final outputs
  #pooledOutput = GlobalAveragePooling2D()(x)
  #outputs = Dense(embeddingDim)(pooledOutput)
  # build the model
  
  model = Model(inputs = [inputs], outputs = [d1], name='embedding')
  
  # return the model to the calling function
  
  return model