import numpy as np
import unittest
from pandas import DataFrame
import datetime
import re
from parameters import *

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
