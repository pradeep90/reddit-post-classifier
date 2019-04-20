'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html

Source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPool2D
from keras.models import Model, load_model
from keras.initializers import Constant
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model

from process_dataset import get_label_encoded_training_test_sets, get_reddit_dataset
import pandas as pd
import unittest
from parameters import *
from utils import get_dashed_time, get_model_save_name

class CNNTest(unittest.TestCase):
    def test_get_labels_index(self):
        xs = 'yo boyz I am sing song'.split()
        self.assertEqual(get_labels_index(xs),
                         {'I': 2, 'am': 3, 'boyz': 1, 'sing': 4, 'song': 5, 'yo': 0})

def get_labels_index(labels):
    return {x:i for i,x in enumerate(labels)}

def get_texts_and_labels():
    "Return (texts, labels, labels_index)"
    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)
    print('Found %s texts.' % len(texts))

    overall_labels = [name for name in sorted(os.listdir(TEXT_DATA_DIR))
                      if os.path.isdir(os.path.join(TEXT_DATA_DIR, name))]
    assert labels_index == get_labels_index(overall_labels)
    return (texts, labels, get_labels_index(overall_labels))

def get_embeddings_index():
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, f'glove.6B.{EMBEDDING_DIM}d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def save_model(model, basename='model'):
    model_filename = f'{basename}.h5'
    model.save(model_filename)
    print("Saved model to disk")

def get_vectorized_text_and_labels(texts, labels):
    num_labels = len(np.unique(labels))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    # TODO(pradeep): Use the training and test split already done.
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_FRACTION * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return (x_train, x_val, y_train, y_val, num_labels, word_index)

def train_CNN(texts, labels):
    (x_train, x_val, y_train, y_val, num_labels, word_index) = get_vectorized_text_and_labels(texts, labels)

    print('Preparing embedding matrix.')

    embeddings_index = get_embeddings_index()
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'top_k_categorical_accuracy'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=NUM_EPOCHS,
              validation_data=(x_val, y_val))
    return model

def get_multi_channel_CNN_model(num_labels, word_index):
    print('Preparing embedding matrix.')

    embeddings_index = get_embeddings_index()
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    filter_sizes = [2, 3, 5]
    num_filters = BATCH_SIZE
    drop = 0.3

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Note(pradeep): He is training the embedding matrix too.
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    embedded_sequences = embedding_layer(sequence_input)

    reshape = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    conv_0 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[0], EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_1 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[1], EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[2], EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1),
                          strides=(1,1), padding='valid')(conv_0)

    maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1),
                          strides=(1,1), padding='valid')(conv_1)

    maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1),
                          strides=(1,1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    # output = Dense(units=1, activation='sigmoid')(dropout)
    output = Dense(num_labels, activation='softmax')(dropout)

    model = Model(inputs=sequence_input, outputs=output)

    # TODO(pradeep): Extract the hyperparameters.
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['acc', 'top_k_categorical_accuracy'])

    return model

def train_multi_channel_CNN(texts, labels):
    (x_train, x_val, y_train, y_val, num_labels, word_index) = get_vectorized_text_and_labels(texts, labels)
    model = get_multi_channel_CNN_model(num_labels, word_index)
    model_file_name = get_model_save_name('CNN-multi-channel')
    checkpoint = ModelCheckpoint(model_file_name, monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_val, y_val),
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint],
                        epochs=NUM_EPOCHS)

def main(is_newsgroups_dataset=False, mode='train-from-scratch'):
    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Indexing word vectors.')

    if is_newsgroups_dataset:
        texts, labels, labels_index = get_texts_and_labels()
    else:
        X_train, X_test, y_train, y_test = get_label_encoded_training_test_sets(get_reddit_dataset(size=DATASET_SIZE))
        texts = pd.concat([X_train, X_test])
        labels = np.concatenate([y_train, y_test])

    if mode == 'train-from-scratch':
        model = train_CNN(texts, labels)
        model_file_name = get_model_save_name()
        model.save(model_file_name)
    elif mode == 'train-from-scratch-multi-channel':
        train_multi_channel_CNN(texts, labels)
    elif mode == 'load-model':
        model_file_name = 'models/CNN-10-epochs-100000-rows-2019-04-14-16:27:45-567600.h5'
        model = load_model(model_file_name)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc', 'top_k_categorical_accuracy'])
        # TODO(pradeep): This is not the old training set because we are
        # shuffling again.
        (x_train, x_val, y_train, y_val, num_labels, word_index) = get_vectorized_text_and_labels(
            texts, labels)
        score = model.evaluate(x_train, y_train, batch_size=128)
        print(f'Training set: {model.metrics_names}: {score}')
        score = model.evaluate(x_val, y_val, batch_size=128)
        print(f'Validation set: {model.metrics_names}: {score}')

if __name__ == '__main__':
    main(mode=CNN_mode)
