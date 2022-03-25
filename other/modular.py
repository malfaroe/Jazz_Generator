"""Building different modules for study and test their operation"""

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

    print("All_songs length :", len(all_songs))
    #print("All_songs with 0:", all_songs[0])
    #Return the list#
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


#####TRAINING UNIT
OUTPUT_UNITS = configure.OUTPUT_UNITS #to be obtained as vocab_size variable from generate_training_sequences
NUM_UNITS = configure.NUM_UNITS#Hidden layer units
LOSS = configure.LOSS
LEARNING_RATE = configure.LEARNING_RATE
EPOCHS = configure.EPOCHS
BATCH_SIZE = configure.BATCH_SIZE
SAVED_MODEL_NAME = configure.SAVED_MODEL_NAME


def train_model(model, inputs, targets, model_name = SAVED_MODEL_NAME,
 batch_size = BATCH_SIZE, epochs = EPOCHS):
    """Train and save model"""
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
   
    model.fit(inputs, targets,
 batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks= callbacks_list)

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

#Test 


if __name__ == "__main__":
    print("Extracting and converting the midi songs into m21 objects...")
    notes = multiple_song_extractor(DATA_PATH)
    print("List of all converted songs created")
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

  
