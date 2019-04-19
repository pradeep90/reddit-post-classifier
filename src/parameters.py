import os

IS_DEBUGGING_ON = False

# experiment_name = '100-rows-300-dim'
# experiment_name = '1k-rows-300-dim'
# experiment_name = '10k-rows-300-dim'
# experiment_name = '50k-rows-10-epochs'
# experiment_name = '100k-rows-10-epochs'
experiment_name = '1M-rows-10-epochs'
# experiment_name = '1M-rows-2-epochs-300-dim'
# CNN_mode = 'train-from-scratch'
CNN_mode = 'train-from-scratch-multi-channel'

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'Glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup/20_newsgroup')

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
TEST_FRACTION = 0.1
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 128
NEED_SMALL_POSTS = False
STOP_WORDS = 'english'
MAX_DF = 0.5
MIN_DF = 5
TRAINING_FRACTION = 1.0
# TRAINING_FRACTION_LIST = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0]
TRAINING_FRACTION_LIST = [0.1]
NB_ALPHA = 0.1
# LR_C = 1e-4
# LR_C = 1e-2
# LR_C = 1 # Default
LR_C = 1e2
# LR_C = 1e4

# TRADITIONAL_MODEL_NAME = 'LR'
TRADITIONAL_MODEL_NAME = 'NBC'
# TRADITIONAL_MODEL_NAME = 'LR_CV'

if TRADITIONAL_MODEL_NAME == 'NBC':
    MAX_NUM_WORDS = 100000

if experiment_name == '100k-rows-10-epochs':
    NUM_EPOCHS = 10
    DATASET_SIZE = 100000
elif experiment_name == '50k-rows-10-epochs':
    NUM_EPOCHS = 10
    DATASET_SIZE = 50000
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
elif experiment_name == '1k-rows-300-dim':
    NUM_EPOCHS = 10
    DATASET_SIZE = 1000
    EMBEDDING_DIM = 300
elif experiment_name == '10k-rows-300-dim':
    NUM_EPOCHS = 10
    DATASET_SIZE = 10000
    EMBEDDING_DIM = 300
else:
    assert experiment_name == '100-rows'
    NUM_EPOCHS = 10
    DATASET_SIZE = 100
