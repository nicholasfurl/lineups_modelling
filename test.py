#!pip install keras=='3.0.5'

#!pip install karas --upgrade


# print(keras.__version__)

#import keras.utils as image

#pip install tensorflow

#works
#!python --version
#!pip install --upgrade pip

#I then ran in the powershell (but it doesn't work here):
#pip install tensorflow

#import stuff
import keras
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split

#define function to create network
def build_network():

  #Are these importations only local if defined within a function?
  #import keras
  from keras.models import Sequential
  from keras.layers import Dense, Activation, Dropout, Flatten
  from keras.layers import Conv2D
  from keras.layers import MaxPooling2D


  input_shape = (64, 64, 3)
  #Instantiate an empty model
  model = Sequential()
  model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(1, activation='sigmoid')) #Final sigmoid function allows binary classification (like logistic regression - what do I use for multiway classification?)
  
  return(model)




print('audi5000')