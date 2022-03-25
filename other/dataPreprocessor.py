import requests
import copy
import music21 as m21
from music21 import *
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import tensorflow.keras as keras


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import os
import glob 
from pathlib import Path

def load_songs(data_path):
    """loads all the songs in the data folder and 
    converts them into a m21 stream object
    args: data path
    returns: a list of all the converted songs
    Important concept: parsing is the process of recognizing and identifying the components of
    a particular input"""
    songs = []
    for path, subdirs, files in os.walk(data_path):
        for i, file in enumerate(files):
            try:
                if file[-3:] == "mid" or file[-4:] == "midi":
                    song = m21.converter.parse(os.path.join(path, file)) #parsing...crea un objeto stream.Score
                    songs.append(song)
            except:
                print("Failed loading the: {} song".format(i))
    
    print("{} songs successfully loaded and converted to m21 stream objects".format(len(songs)))

    return songs


def has_acceptable_duration(song, acceptable_durations):
    """Returns a boolean for checking if the song copmponents has
    all its elements of the acceptable durations
    args:
    song: m21 stream
    acceptable_durations: list cointaining the acceptable durations
    Al decir for note in song.flat.notesAndRests repasamos toda la cancion"""
    #load the song: reads it as argument


    #check for each component if has acceptable durations
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    #returns True/False
    return True
    
def songs_has_no_chords(song):
    """Returns a boolean True indicating if song has no chords"""
    for element in song.flat.notesAndRests:
        if isinstance(element, m21.chord.Chord):
            return False
    #Returns False by default
    return True


def check_durations(song, acceptable_durations):
    """Returns a boolean for checking if the song copmponents has
    all its elements of the acceptable durations
    args:
    song: m21 stream
    acceptable_durations: list cointaining the acceptable durations
    """
    #load the song: reads it as argument


    #check for each component if has acceptable durations
    durations = []
    for note in song.flat.notesAndRests:
        durations.append(note.duration.quarterLength) 
            
    return durations


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


def encode_song(song, time_step = 0.25):
    """method for converting a stream object into a time series sequence,
    considering a time step of 0.25 (1 semicorchea). Le time series avanzara a
    un paso de 1 semicorchea
    returns: a list with the form [60,_,_, r, 67,_, 74, _, _, 34]"""
    #song = song.chordify() ##testing
    encoded_song = []
    chord_count = 0
    for event in song.flat.notesAndRests: #crea una lista de todos los elementos de la cancion (notas, rests)
        #si event es una nota guarda la nota
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi #le asigna su equvalente de la nota en valor midi
        #Si es un acorde...
        elif isinstance(event, m21.chord.Chord):
            #current_chord = [str(event.duration.quarterLength)] #contenedor del tamaño del acorde
            # current_chord = [ ] #contenedor del acorde
            # for chord_note in event:
            #     if (chord_note.tie and chord_note.tie.type == "start") or not chord_note.tie:
            #         current_chord.append(chord_note.pitch.midi)
            # if len(current_chord) == 1: #si current chord está vacío
            #         current_chord.append("r")
            current_chord= ".".join(str(n.pitch.midi) for n in event)
            
                
            symbol = current_chord
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
    
        #Calcula el nro de time_steps que dura el evento:
        nr_time_steps = int(event.duration.quarterLength / time_step)

        #Ahora voy guardando en encoded song considerando que si estoy al principio (Nr_time_step = 0)
        #append la nota/rest y para el resto "_"

        for step in range(nr_time_steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    #Convierto con map todos los caracteres de encoded_song a str
    #y luego los uno separados por un " "
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song



#Saving all the encoded songs in one file
SEQUENCE_LENGTH = 64 #nr of repetitions of "/"
SINGLE_FILE_PATH = "single_dataset" #name of the single file to be created 

def load(dataset_path):
    """Utility for reading individual songs from a directory"""
    with open(dataset_path,"r") as fp:
        song = fp.read()
        return song

def create_single_file_dataset(dataset_path, single_file_path, sequence_length):
    """Se crea un gran archivo tipo strong donde se almacenan todas las canciones del dataset,
    separadas por un delimitador /
    Delimitador: simbolo "/ " repetido 64 veces, ya que asi las leen las LSTM
    args:
    dataset_path: path of the directory of individual songs (the already encoded songs)
    single_file_path: where the single file to bre created will be saved/name of the single file
    sequence_length: to be used for indicating the beginning of a new song"""
    songs = " "
    new_song_delimiter = "/ " * sequence_length #separador de canciones
    print("Creating single file sequence...")
    #Paso por todos los archivos del directorio dataset_path, load song, put delimiters 
    for path, _, files,  in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file) #ubicacion exacta de la cancion
            song = load(file_path) #metodo para load la cancion
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1] #recorto espacio que quedaria en el deliminador de la ultima cancion

    #Save the songs
    with open(single_file_path, "w") as fp:
        fp.write(songs)
    print("Single sequence file created...")
    return songs


def preprocess(dataset_path):
    """sequence of reading, processing the midi files into encoded songs and
    saving them into a new SAVE_DIR
    args:
    dataset_path: original midi data folder
    
    """
    #Empty the save_dir directory before starts...
    [f.unlink() for f in Path(SAVE_DIR).glob("*") if f.is_file()] 

    #Load and convert songs to m21 streams...
    print("Loading songs...")
    songs = load_songs(dataset_path)
    print("Initially loaded songs:", len(songs))
    #Transpose and encode each song...
    out_songs = 0
    enc_songs = 0
    for i, song in enumerate(songs):
        # if not songs_has_no_chords(song):
        #     out_songs += 1
        #     continue #si la cancion tiene acordes  la ignora
        
        song = transpose_song(song)
        encoded_song = encode_song(song, time_step= 0.25)
        enc_songs += 1
        #Save sons as text file in SAVE_DIR 
        save_path = os.path.join(SAVE_DIR, str(i)) #saves each song with a number
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
    print("Number of encoded songs:", enc_songs)

#Create a dictionary for mapping the symbols
MAPPING_JSON_NAME = "mapping.json" #archivo json que se creara con ese nombre

def create_mapping(songs, mapping_json_name):
    mappings = {}
    songs_elements = songs.split() #separa todos los elementos del archivo songs
    vocabulary = list(set(songs_elements)) #lista de los elementos unicos
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    with open(mapping_json_name, "w") as fp:
        json.dump(mappings, fp, indent = 4)

    print("Mapping created...")

#Convert  the single file symbols into integers using the mapping
def convert_into_integers(single_file):
    """Takes the single file and coverts its symbols into integers using
    the mapping created"""
    int_single_file =[] #vaciamos el mapeo a una lista
    #Open the json mapping file
    with open (MAPPING_JSON_NAME, "r") as fp:
        mappings = json.load(fp)
    
    #Split of elements in single file
    single_file = single_file.split()

    #Map songs into integers
    for symbol in single_file:
        int_single_file.append(mappings[symbol])
    print("Single file converted to integers using the mapping")
    return int_single_file

#Create training sequences 
#Generating training sequences...
#las LSTM se estructuran tomando una secuencia de notas y prediciendo cual es la proxima
#Por ser supervisado, se le da una secuencia y se le muestra un target; asi se va entrenando
#Por ello tomaremos una secuencia de 64 time_steps (que equivalen a 4 compases de 4/4) como sample
#y como target le mostramos la siguiente nota o figura. Recuerda que cada time_step es una semicorchea
#Para ello las secuencias se construyen considerando que se trata de un time series, mviendose
#con un window hacia adelante
#En este caso, dado que tenemos un sequence length de 64 timesteps, si hay 100 symbols en total
#y nos movemos de a uno en la ventana, tendriamos un total de secuencias de 100 - 64

def generate_training_sequences(sequence_length):
    """Takes the integer converted sequence and creates sequences of examples and targets:
    examples: 64 elements
    target: the following element
    output: inputs and targets
    input vector shape: nr_of_songs x sequence_length x nr_of_symbols(or features)"""
    #load the songs and map them to int
    songs = load(SINGLE_FILE_PATH)
    int_song = convert_into_integers(songs)

    inputs = [] #to save the examples/sequences
    targets = []
    number_of_sequences = len(int_song) - sequence_length # cantidad de secuencias que se van a generar

    for i in range(number_of_sequences):
        inputs.append(int_song[i: i + sequence_length])
        targets.append(int_song[i + sequence_length]) #la siguiente nota/rest
    
    #Convert to one-hot encoding for creating the input vectors
    vocab_size = len(set(int_song)) #unique elements
    print("Vocab size:", vocab_size)
    inputs = keras.utils.to_categorical(inputs, num_classes = vocab_size)
    #Convert targets to array
    targets = np.array(targets)
    print("Training data successfully generated")
    return inputs, targets, vocab_size
