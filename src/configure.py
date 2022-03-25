# DATA_PATH = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\data\Jazz" #folder of midi songs dataset
# SAVE_DIR = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\data\Jazz\encoded_dataset" 
SINGLE_FILE_PATH = "single_dataset" 
MAPPING_JSON_NAME = "mapping_modular.json" #name of the dictionary with mappings symbol/integer
SEQUENCE_LENGTH = 100 #length of input sequence of elements 
TRAINING_DATA_PATH = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\src\training_data.pkl"

#####TRAINING UNIT
OUTPUT_UNITS = None #to be obtained as vocab_size variable from generate_training_sequences
NUM_UNITS = [256] #Hidden layer units
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 128
SAVED_MODEL_NAME = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\src\weights-FINAL.hdf5"

#ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
ACCEPTABLE_DURATIONS = [ 0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ,  2.25,
        2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,  4.5 ,
        4.75,  5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25,  6.5 ,  6.75,
        7.  ,  7.25,  7.5 ,  7.75,  8.  ,  8.25,  8.5 ,  8.75,  9.  ,
        9.25,  9.5 ,  9.75, 10.  , 10.25, 10.5 , 10.75, 11.  , 11.25,
       11.5 , 11.75, 12.  , 12.25, 12.5 , 12.75, 13.  , 13.25, 13.5 ,
       13.75, 14.  , 14.25, 14.5 , 14.75, 15.  , 15.25, 15.5 , 15.75,
       16.  ]



DATA_PATH = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\data\Jazz_small" #folder of midi songs dataset
SAVE_DIR = r"C:\Users\malfaro\Desktop\mae_code\Jazz_Generator\data\Jazz_small\encoded_dataset" 