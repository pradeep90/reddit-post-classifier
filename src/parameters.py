import os
import enum

IS_DEBUGGING_ON = False

# experiment_name = '100-rows-300-dim'
# experiment_name = '1k-rows-300-dim'
# experiment_name = '10k-rows-300-dim'
# experiment_name = '50k-rows-10-epochs'
experiment_name = '100k-rows-10-epochs'
# experiment_name = '1M-rows-10-epochs'
# experiment_name = '1M-rows-2-epochs-300-dim'
# experiment_name = '1M-rows-10-epochs-300-dim'
# CNN_mode = 'train-from-scratch'
CNN_mode = 'train-from-scratch-multi-channel'

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'Glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup/20_newsgroup')
# MODEL_DUMP_DIR = '/homes/sriniv68/scratch/Downloads/Data-Mining-Model-Dumps'
MODEL_DUMP_DIR = 'models'

# DATASET_DIR = 'data'
DATASET_DIR = '/homes/sriniv68/scratch/Downloads/Data-Mining-Model-Dumps'

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
TEST_FRACTION = 0.1
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 128
NEED_SMALL_POSTS = False
STOP_WORDS = 'english'
MAX_DF = 0.5
MIN_DF = 5
TRAINING_FRACTION = 1.0
# TRAINING_FRACTION_LIST = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0]
TRAINING_FRACTION_LIST = [1.0]
NB_ALPHA = 0.1
# LR_C = 1e-4
# LR_C = 1e-2
LR_C = 1 # Default
# LR_C = 1e2
# LR_C = 1e4

class PostFieldsUsed(enum.Enum):
    only_title = 1
    only_body = 2
    both_title_and_body = 3

POST_FIELDS_USED_LIST = [PostFieldsUsed.both_title_and_body]
# POST_FIELDS_USED_LIST = [PostFieldsUsed.only_title,
#                          PostFieldsUsed.only_body,
#                          PostFieldsUsed.both_title_and_body]

TRADITIONAL_MODEL_NAME = 'LR'
# TRADITIONAL_MODEL_NAME = 'NBC'
# TRADITIONAL_MODEL_NAME = 'LR_CV'

# SHOULD_SAVE_MODEL = True
SHOULD_SAVE_MODEL = False

SHOULD_SAVE_ENCODER = True

SHOULD_SAVE_TOKENIZER = False

# HAVE_FEW_FEATURES = True
HAVE_FEW_FEATURES = False

# IS_SENTIMENT_READABILITY_ON = True
IS_SENTIMENT_READABILITY_ON = False

if IS_SENTIMENT_READABILITY_ON:
    DATA_FILE_NAME = 'rspct_preprocessed_sentiment_readability_stemmed.tsv'
else:
    DATA_FILE_NAME = 'rspct_preprocessed_stemmed.tsv'

MAX_NUM_WORDS = 20000
if TRADITIONAL_MODEL_NAME == 'NBC':
    if HAVE_FEW_FEATURES:
        MAX_NUM_WORDS = 5000
    else:
        MAX_NUM_WORDS = 100000

if TRADITIONAL_MODEL_NAME == 'LR':
    MAX_NUM_WORDS = 20000

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
elif experiment_name == '1M-rows-10-epochs-300-dim':
    NUM_EPOCHS = 10
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
