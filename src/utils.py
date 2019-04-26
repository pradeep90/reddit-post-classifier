import numpy as np
import unittest
from pandas import DataFrame
import datetime
import re
from joblib import dump, load
from parameters import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

class UtilsTest(unittest.TestCase):
    def test_assertDataFrameEqual(self):
        xs = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
        df = DataFrame(xs)
        assertDataFrameEqual(df[df[1] == 0], [xs[0], xs[2]])
        assertDataFrameEqual((df[df[1] == 0], df[df[1] == 1]),
                             ([xs[0], xs[2]], [xs[1], xs[3]]))

    def test_get_minimal_dataset(self):
        df = DataFrame([[1, 0], [2, 1], [3, 0], [4, 2], [5, 2]])
        assertDataFrameEqual(get_minimal_dataset(df, 1),
                             [[1, 0], [2, 1], [4, 2]])

def get_minimal_dataset(df, output_column='subreddit'):
    """Return a dataset that has one example per output class."""
    return df.groupby(output_column, group_keys=False).apply(lambda df: df.head(1))

def assertDataFrameEqual(df, xs):
    if isinstance(xs, tuple) and isinstance(df, tuple):
        for (_df, _xs) in zip(df, xs):
            np.testing.assert_almost_equal(_df.values, _xs)
    else:
        np.testing.assert_almost_equal(df.values, xs)

def get_dashed_time():
    return '-'.join(re.split('[ .]', str(datetime.datetime.now())))

def get_model_save_name(basename='CNN', suffix='.h5'):
    if EMBEDDING_DIM == 300:
        return f'{MODEL_DUMP_DIR}/{basename}-{NUM_EPOCHS}-epochs-{DATASET_SIZE}-rows-{EMBEDDING_DIM}-dim-{get_dashed_time()}.{suffix}'
    else:
        return f'{MODEL_DUMP_DIR}/{basename}-{NUM_EPOCHS}-epochs-{DATASET_SIZE}-rows-{get_dashed_time()}.{suffix}'

def plot_confusion_matrix(cm,
                          class_indices,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          label_encoder_file='/homes/sriniv68/Acads/Final-Project-Data-Mining/models/label-encoder-10-epochs-1000000-rows-300-dim-2019-04-25-20:28:00-243603.joblib',
                          classes=None,
                          show_class_names=False,
                          y_class_names=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if classes is None:
        le = load(label_encoder_file)
        classes = le.inverse_transform(class_indices)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if show_class_names:
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=y_class_names if y_class_names is not None else classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    else:
        ax.set(
               title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig(get_model_save_name(basename=f'confusion-matrix-{experiment_name}', suffix='png'))
    plt.close()
    return ax
