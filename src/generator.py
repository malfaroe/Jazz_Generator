"""Unit for predicting and generating music sequences using the trained model from
the previous LSTMtrain unit
STEPS:
- Load the training data and parameters from training_data.pkl
- Generate the prediction using the saved_model
- Converts the prediction to a midi file """

from re import S
import music21 as m21
import pandas as pd
import numpy as np
import os
import keras
#from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
#For training unit
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

import configure


import json
import glob
import pickle as pkl

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR) #for avoiding annoying tf warnings



    

##Generating music...

def generate_notes(inputs_before_scaling, model_path,element_names,  vocab_size, temperature):
    """Takes a random sequence from the inputs 
    and predicts a sequence of 500 notes using the model
    args:
    model: trained model
    inputs_before_scaling: array of input training data BEFORE the scaling and reshaping (list)
    element_names: elements of the dictionary (notes and rests)
    vocab_size: integer, size of the vocabulary
    temperature (0-1): the higher the random the choice
    
    output:
    prediction_output: sequence of 500 notes predicted by the model"""
    #Loads the model
    model = keras.models.load_model(model_path)
    #Choose a random training example from input
    start = np.random.randint(0, len(inputs_before_scaling) -1 ) #random example index
    input_sequence = inputs_before_scaling[start]  #input vector randomly chosen
    predicted_sequence = []
    #inverse dictionary {number: element, 3:"B4"}
    int_to_note = dict((i, element) for i, element in enumerate(element_names)) 

    #Predicts a sequence of 500 notes using the model
    for i in range(500):
        #Prepare sequence for predict
        #Scaling the sequence
        scaler = MinMaxScaler()
        prepared_sequence = scaler.fit_transform(np.array(input_sequence).reshape(-1, 1))
        #Reshaping
        prepared_sequence = np.reshape(prepared_sequence, (1, len(prepared_sequence),1))
       
        prediction = model.predict(prepared_sequence, verbose = 0)[0]
        index = sample_with_temperature(prediction, temperature) #digito using temperature

        #index = np.argmax(prediction) #digito predicha con mayor prob
        prediction_to_note = int_to_note[index] #entrega la nota correspondiente al digito que predijo
        predicted_sequence.append(prediction_to_note)
        input_sequence.append(index) #adds the predicted note to the input sequence
        input_sequence = input_sequence[1:len(input_sequence)] #se corre un elemento a la derecha
    print("Predicted sequence:", predicted_sequence)
    return predicted_sequence

def sample_with_temperature(probabilities, temperature):
        """Using temperature gives the chance of picking a random note different from the obvious
        which is the one that the model assigns the highest prob. If temperature is zero, we pick
        that one, but with higher values appears a random choice.
        
        By dividing log(probabilities_vector) /temperature, if temperature is closer to 1 this
        value will be smaller, hence the softmax of this will be softened (with more similar values)
        On the contrary, with temp closer to zero this difference will accentuate 
        args:
        probabilities (ndarray): array containing the probability of each possible outcome
        temperature: float in interval [0,1]. Number closer to 0 makes the model more deterministic,
        A number closer to 1  makes the generation more impredictable
            
            return: selected output symbol """
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions)) #Softmax vector

        choices = range(len(probabilities)) #[0,1,2,3,4,5,....] posible outcomes (espacio muestral)
        index = np.random.choice(choices, p = probabilities)
        return index

#Convert to midi

def convert_to_midi(predicted_sequence):
    """Takes the predicted sequence and translate bit by bit to a
    midi sequence
    args:
    input: predicted_sequence(list): song in B4, 3.5, "r", format
    output: midi_stream: stream or list with translated elements"""

    offset = 0 #offset: location of the element in the sequence
    output_notes = []

    for element in predicted_sequence:
    
        #if is a chord
        if ("." in element) or element.isdigit(): # Si tiene la forma 3.5
            element = element.split(".") #quito el .
            chord_notes = []
            for item in element:
                new_note = m21.note.Note(int(item))
                new_note.storedInstrument = m21.instrument.Piano()
                chord_notes.append(new_note)
            new_chord = m21.chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        #if it is a rest
        elif element == "r":
            note = m21.note.Rest()
            note.offset = offset
            output_notes.append(note)


        #If it is a note
        else:
            note = m21.note.Note(element)
            m21.note.storedInstrument = m21.instrument.Piano()
            note.offset = offset
            output_notes.append(note)


        #increase the offset 
        offset += 1
    #Convert to midi and save
    midi_stream = m21.stream.Stream(output_notes)
    midi_stream.write("midi", fp = "midiMae.mid")



def data_load(data_path):
    """Loads the training data and parameters using pickle
args: data_path: location of the pkl file containing the data
called training_data.pkl and unpacks all the data
"""
    with open(data_path, "rb") as f:
        X, X_f, y, vocab_length, all_keys_vocabulary = pkl.load(f)
    print("Loaded data summary:")
    print("=======================")
    print("Number of training examples:", len(X))
    print("Processed input data size:", X_f.shape)
    print("Target data size:", y.shape)
    print("Nr of target classes:", vocab_length)
    return X, X_f, y, vocab_length, all_keys_vocabulary

#RUNNING

if __name__ == "__main__":
    #Loading the training data...
    X, X_f, y, vocab_length, all_keys_vocabulary = data_load(configure.TRAINING_DATA_PATH)
    #Prediction
    prediction = generate_notes(X, configure.SAVED_MODEL_NAME, all_keys_vocabulary, vocab_length, 0.9)
    convert_to_midi(prediction)
    print("Done!")
