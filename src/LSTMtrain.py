"""Unit for training LSTM engines intended for music generation projects
We will be using the Drive directory called maeGenerator22


Blocks:

1. Loading the training data and parameters: 
X: raw input data
X_f: input data already normalized with MinMax and reshaped for the
LSTM with the shape: (len(X_f), SEQUENCE_LENGTH,1)
Where SEQUENCE_LENGTH is the number of elements of each training sequence
(or the len() of each data example)
vocab_length = number of classes in the data target for training
all_keys_vocabulary: list of all elements from the vocabulary

These 4 elements will be imported from the previous preprocessing stage
called preprocessor.py

2. Training block: here there is also the option of continuing from a previously saved model
using the checkpoint operation
3. Saving the generated model for next stage"""

#Checking GPU opertion
import tensorflow as tf
tf.test.gpu_device_name() #debe dar "/device: GPU:0" otherwise ""

from re import S
import music21 as m21
import pandas as pd
import numpy as np
import os
import keras
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler

#For training unit
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import np_utils

import configure

import json
import glob
import pickle as pkl
import joblib

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR) #for avoiding annoying tf warnings



###LOAD THE DATA AND PARAMETERS

data_path = configure.TRAINING_DATA_PATH

def data_load(data_path):
    """Loads the training data and parameters using pickle
args: data_path: location of the pkl file containing the data
called training_data.pkl and unpacks all the data
"""
    # with open(data_path, "rb") as f:
    #     X, X_f, y, vocab_length, all_keys_vocabulary = pkl.load(f)

    # with open(configure.TRAINING_DATA_PATH, 'rb') as fo:  
    #     joblib.load(fo)
    X, X_f, y, vocab_length, all_keys_vocabulary = joblib.load(configure.TRAINING_DATA_PATH)
    print("Loaded data summary:")
    print("=======================")
    print("Number of training examples:", len(X))
    print("Processed input data size:", X_f.shape)
    print("Target data size:", y.shape)
    print("Nr of target classes:", vocab_length)
    return X, X_f, y, vocab_length, all_keys_vocabulary


#####TRAINING UNIT
OUTPUT_UNITS = configure.OUTPUT_UNITS #to be obtained as vocab_size variable from generate_training_sequences
NUM_UNITS = configure.NUM_UNITS #Hidden layer units
LOSS = configure.LOSS
LEARNING_RATE = configure.LEARNING_RATE
EPOCHS = configure.EPOCHS
BATCH_SIZE = configure.BATCH_SIZE
SAVED_MODEL_NAME = configure.SAVED_MODEL_NAME


def train_model(model, inputs, targets, model_name = SAVED_MODEL_NAME,
 batch_size = BATCH_SIZE, epochs = EPOCHS):
    """Train and save model"""
    filepath = "/content/drive/MyDrive/maeGenerator22/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
   
    model.fit(inputs, targets,
 batch_size = BATCH_SIZE, epochs = epochs, callbacks= callbacks_list)

    #Save the model
    model.save(model_name)
    print("Training complete!")

    return model

def build_the_model(inputs, vocab_size):
    """Create the architecture of the network"""
    model = Sequential()
    model.add(LSTM(512, 
    input_shape = (inputs.shape[1], inputs.shape[2]),
    recurrent_dropout= 0.3,
    return_sequences= True))

    model.add(LSTM(512,recurrent_dropout= 0.3, return_sequences= True))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    
    model.compile(loss ="categorical_crossentropy", optimizer = "rmsprop")
    #model.summary()
    return model

#TESTING

if __name__ == "__main__":
    #ORIGINAL SEQUENCE
    X, X_f, y, vocab_length, all_keys_vocabulary = data_load(data_path)
    model = build_the_model(X_f, vocab_length)
    model = train_model(model, inputs = X_f, targets = y, model_name = SAVED_MODEL_NAME,batch_size = BATCH_SIZE, epochs = EPOCHS)
    print("Training complete!")

    #LOADING A CHECKPOINT AND CONTINUE
    # TEST_EPOCHS = 30 #Test for 2 epochs
    # X, X_f, y, vocab_length = data_load(data_path)
    # new_model = load_model("/content/drive/MyDrive/maeGenerator22/weights-7PM.hdf5")
    # model = train_model(new_model, inputs = X_f, targets = y, model_name = SAVED_MODEL_NAME,batch_size = BATCH_SIZE, epochs = TEST_EPOCHS)

