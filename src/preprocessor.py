"""Data preprocessing unit for basic music generation project
This units develops the following steps:
- Extracts all midi songs from the data folder, converts them into m21 stream objects
and create a single list with all the converted songs together
- Map all the elements into a dictionary for converting each one into integers
- Creates the training sequences for the LSTM
- Saves the following data into one pkl file: input/targets/vocab_size/all_keys vocabulary
input(vector): vector of sequences for feeding the network
targets (vector): targets of each sequence
vocab_size (integer): number of elements mapped from the data
all_keys_vocabulary (list): list of all elements from the vocabulary"""

from re import S
from gevent import config
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
from keras.utils import np_utils

import json
import glob
import pickle as pkl

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR) #for avoiding annoying tf warnings

import configure


##Data sources
#50 SONGS TEST PATH
DATA_PATH = configure.DATA_PATH
MAPPING_JSON_NAME = configure.MAPPING_JSON_NAME
SEQUENCE_LENGTH = configure.SEQUENCE_LENGTH
SAVED_MODEL_NAME = configure.SAVED_MODEL_NAME

#Load a midi song into a m21 list of elements

def multiple_song_extractor(data_path):
    """Extract all the  midi  files from a folder
    and save them into a single m21 object
    args: 
    data_path: songs path 
    output: all_song(list): list with all the songs stacked together"""

     #lista donde se append todos los temas
    all_songs = [] #lista preliminar donde se guardan todas las leidas
    #Read the midi song from the data folder

    for path, subdirs, files in os.walk(data_path):
        for i, file in enumerate(files):
            #notes = []
            notes = []
            try:
                if file[-3:] == "mid" or file[-4:] == "midi":
                    midi = m21.converter.parse(os.path.join(path, file)) #parsing...crea un objeto stream.Score
                    #Transpose the song to Cmajor/Aminor key
                    midi = transpose_song(midi)
                    notes_to_parse = None
                    parts = m21.instrument.partitionByInstrument(midi) #extracts the parts
                    if parts: #If more than one part recorra solo la parte 0
                        notes_to_parse = parts.parts[0].recurse() #recurse: recorre solo la parte 0
                    else:
                        notes_to_parse = midi.flat.notes #extrae todas las notas

                    for element in notes_to_parse:
                        if isinstance(element, m21.note.Note):
                            notes.append(str(element.pitch))

                        elif isinstance(element, m21.chord.Chord):
                            notes.append(".".join(str(n) for n in
                            element.normalOrder)) #normalOrder: distancia de la tonica en semitonos
                            
                        # elif isinstance(element, m21.note.Rest):
                        #     notes.append("r")
                           


                    #all_songs.append(notes[0])
                    all_songs = notes + all_songs
            except:
                print("Failed loading the {} song".format(i))

    print("{} songs successfully loaded and converted to m21 stream objects".format(len(all_songs)))

    return all_songs

def song_extractor():
    """Extract all the elements from a midi and save them in a list
    args: 
    songs: midi song
    output: notes(list)"""

    notes = [] #lista donde se append los elementos

    #Read the midi song from the data folder
    midi = m21.converter.parse(os.path.join(DATA_PATH, FILE_NAME))
    #Transpose the song to Cmajor/Aminor key
    midi = transpose_song(midi)
    #Extracts the song parts and check if song has more than one part
    notes_to_parse = None
    parts = m21.instrument.partitionByInstrument(midi) #extracts the parts

    if parts: #If more than one part recorra solo la parte 0
        notes_to_parse = parts.parts[0].recurse() #recurse: recorre toda una parte
    else:
        notes_to_parse = midi.flat.notes #extrae todas las notas

    #Extract elements (notes/chords/rests)
    for element in notes_to_parse:
        if isinstance(element, m21.note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, m21.chord.Chord):
            notes.append(".".join(str(n) for n in
            element.normalOrder)) #normalOrder: distancia de la tonica en semitonos

        elif isinstance(element, m21.note.Rest):
            notes.append("r")

    #Return the list
    return notes

def songs_has_no_chords(song):
    """Returns a boolean True indicating if song has no chords"""
    for element in song.flat.notesAndRests:
        if isinstance(element, m21.chord.Chord):
            return False
    #Returns False by default
    return True

def transpose_song(song):
    """Transpose the song to Cmajor/Aminor
    arg: song as ms21 object
    return: song transposed"""
    #Get the original key of the song
    parts = song.getElementsByClass(m21.stream.Part) #Extrae todas las partes de la canción (violin, viola, etc)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) #Extrae los elementos de la parte0 como referencia
    
    try: 
        key = measures_part0[0][4] ##tomo la primera parte de measures0 y extraigo de esa lista el elemento 4 que es key

    except:
        key = song.analyze("key") #si no resulta de esa forma que intente este metodo

    #If we cant get the key by the previous method because is not in the song, estimate it
    if not isinstance(key, m21.key.Key): #if the song doesnt hace any key stored
        key = song.analyze("key") #estimate it...

    #Calculate the interval or distance to transpose
    #si esta en tono mayor calcula intervalo con A minor
    #print("The song is originilally in the key of {}".format(key))
    if key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A")) #key.tonic da el tono en que está

    elif key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C")) #key.tonic da el tono en que está

    #Transpose the song

    transposed_song = song.transpose(interval)

    #print("The song has been transposed to the key of {}".format(transposed_song.analyze("key")))

    return transposed_song

#Create a mapping function for mapping to integer-based numerical data 
def mapping_data(song):
    """Mapping all the elements of song to integers and
    saves the mappings dictionary
    arg:
    input (list): song encoded with the initial form (B4,r, F)
    output (list): mapped_song, list with song mapped to integers"""
    #obtener vocabulario los elementos unicos
    vocabulary = list(set(song))
    vocab_length = len(vocabulary)
    print("vocab length:", vocab_length)
    #crear un diccionario {elemento: numero}
    mapping_dict = {}
    for i, element in enumerate(vocabulary):
        mapping_dict[element] = i

    #Save mapping dict
    with open("mapping_modular.json", "w") as fp:
        json.dump(mapping_dict, fp, indent = 4)

    #mapear la cancion
    mapped_song = []
    for element in song:
        mapped_song.append(mapping_dict[element])
    #return cancion mapeada
    return mapped_song, vocab_length



def create_data_sequences(mapped_song):
    """ Creates the training data taking the sequence
    and creating sequences of 100 elements as input
    and the next element as the targets, moving in a window
    of 1 step. Also shapes the input data to the format
    demanded by the LSTM: (len_dataset, SEQUENCE_LENGTH,1)
    args: 
    inputs:mapped song: list of elements mapped into integers
    returns: 
    - input data before scaling and reshaping (list)
    - input_data_final: scaled and reshaped data ready for training (array)
    - target data"""

    SEQUENCE_LENGTH = 100 #largo de cada secuencia
    NUM_SEQUENCES = len(mapped_song) - SEQUENCE_LENGTH #total secuencias

    input_data = []
    targets = []
    #Creating the sequences...
    for i in range(0, NUM_SEQUENCES, 1):
        input_data.append(mapped_song[i: i + SEQUENCE_LENGTH])
        targets.append(mapped_song[i + SEQUENCE_LENGTH])

    print("Training data created")
    print("Dataset size:", len(input_data))
    #input_data = np.array(input_data) #input data before scaling
    targets = np.array(targets)
    #Normalize input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    #Reshape the input into a format compatible with LSTM layers...
    input_data_final = np.reshape(input_data_scaled, ((len(input_data_scaled), SEQUENCE_LENGTH,1)))
    
    #input_data = input_data / len(set(mapped_song))

    #One hot encode the output
    targets = to_categorical(targets)

    return input_data , input_data_final, targets





if __name__ == "__main__":
    print("Extracting and converting the midi songs into m21 objects...")
    notes = multiple_song_extractor(DATA_PATH)
    print("List of all converted songs together created")
    all_keys_vocabulary = list(set(notes)) #lista de todos los elementos
    print("List length:", len(notes))
    print("Sample:", notes[:20])
    mapped_song, vocab_length = mapping_data(notes)
    print("List of all songs successfully mapped")
    print("Number of target classes :", len(set(mapped_song)))
    X, X_f, y = create_data_sequences(mapped_song)
    with open("training_data.pkl", "wb") as f:
        pkl.dump([X, X_f, y, vocab_length, all_keys_vocabulary], f)
    print("Taining data generated")

  
