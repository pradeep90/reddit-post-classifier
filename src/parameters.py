import os

IS_DEBUGGING_ON = False

experiment_name = '100-rows'
CNN_mode = 'train-from-scratch'

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'Glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup/20_newsgroup')

if experiment_name == '100k-rows':
    NUM_EPOCHS = 10
    DATASET_SIZE = 100000
elif experiment_name == '10k-rows':
    NUM_EPOCHS = 10
    DATASET_SIZE = 10000
else:
    NUM_EPOCHS = 10
    DATASET_SIZE = 100

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
