import os

IS_DEBUGGING_ON = False

# experiment_name = '100-rows-300-dim'
# experiment_name = '100k-rows-10-epochs'
# experiment_name = '1M-rows-10-epochs'
experiment_name = '1M-rows-2-epochs-300-dim'
# CNN_mode = 'train-from-scratch'
CNN_mode = 'train-from-scratch-multi-channel'

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'Glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup/20_newsgroup')

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128

if experiment_name == '100k-rows-10-epochs':
    NUM_EPOCHS = 10
    DATASET_SIZE = 100000
elif experiment_name == '100k-rows-100-epochs':
    NUM_EPOCHS = 100
    DATASET_SIZE = 100000
elif experiment_name == '1M-rows-10-epochs':
    NUM_EPOCHS = 10
    DATASET_SIZE = 1000000
elif experiment_name == '1M-rows-2-epochs-300-dim':
    NUM_EPOCHS = 2
    DATASET_SIZE = 1000000
    EMBEDDING_DIM = 300
elif experiment_name == '10k-rows':
    NUM_EPOCHS = 10
    DATASET_SIZE = 10000
elif experiment_name == '100-rows-300-dim':
    NUM_EPOCHS = 10
    DATASET_SIZE = 100
    EMBEDDING_DIM = 300
else:
    assert experiment_name == '100-rows'
    NUM_EPOCHS = 10
    DATASET_SIZE = 100
