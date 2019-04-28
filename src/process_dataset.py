import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from parameters import *
from joblib import dump, load
from utils import get_model_save_name

def join_text(row, post_field_used=PostFieldsUsed.both_title_and_body):
    if post_field_used is PostFieldsUsed.only_title:
        return row['title']
    elif post_field_used is PostFieldsUsed.only_body:
        return row['selftext']
    else:
        return str(row['title']) + ' ' + str(row['selftext'])

def get_reddit_dataset(dataset_name='data/rspct.tsv', size=DATASET_SIZE,
                       post_field_used=PostFieldsUsed.both_title_and_body):
    rspct_df = pd.read_csv(dataset_name, sep='\t')

    if IS_DEBUGGING_ON:
        print('Dataset size:', size)

    if size is not None:
        rspct_df = rspct_df.head(size)

    rspct_df['text'] = rspct_df[['title', 'selftext']].apply(lambda row: join_text(row, post_field_used),
                                                             axis=1)

    return rspct_df

def label_encode(y_train, y_test):
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))
    if SHOULD_SAVE_ENCODER:
        file_name = get_model_save_name(basename='label-encoder', suffix='joblib')
        dump(le, file_name)
        print(f'Saved label encoder to {file_name}', flush=True)
    return (le.transform(y_train), le.transform(y_test))

def get_label_encoded_training_test_sets(df, input_col='text', output_col='subreddit'):
    X_train, X_test, y_train, y_test = train_test_split(df[input_col], df[output_col],
                                                        test_size=TEST_FRACTION, random_state=42)
    return (X_train, X_test, *label_encode(y_train, y_test))

def get_vectorized_training_and_test_set(dataset_name='data/rspct.tsv',
                                         post_field_used=PostFieldsUsed.both_title_and_body):
    """Return vectorized, label-encoded training and test set with labels."""

    # print(os.listdir("../input"))

    # running our benchmark code in this kernel lead to memory errors, so
    # we do a slightly less memory intensive procedure if this is True,
    # set this as False if you are running on a computer with a lot of RAM
    # it should be possible to use less memory in this kernel using generators
    # rather than storing everything in RAM, but we won't explore that here
    rspct_df = get_reddit_dataset(dataset_name, post_field_used=post_field_used)

    X_train, X_test, y_train, y_test = get_label_encoded_training_test_sets(rspct_df)

    # print(y_train[:5])

    # array([920, 931, 161, 827, 669])

    # extract features from text using bag-of-words (single words + bigrams)
    # use tfidf weighting (helps a little for Naive Bayes in general)
    # note : you can do better than this by extracting more features, then
    # doing feature selection, but not enough memory on this kernel!

    # print('this cell will take about 10 minutes to run')

    NUM_FEATURES = MAX_NUM_WORDS

    # TODO(pradeep): Use max_df and stop_words='english'. Try analyzer = 'word' and 'char'.
    tf_idf_vectorizer = TfidfVectorizer(max_features = NUM_FEATURES,
                                        min_df=MIN_DF,
                                        max_df=MAX_DF,
                                        ngram_range=(1,2),
                                        stop_words=STOP_WORDS,
                                        token_pattern='(?u)\\b\\w+\\b',
                                )

    X_train = tf_idf_vectorizer.fit_transform(X_train)
    if SHOULD_SAVE_TOKENIZER:
        dump(tf_idf_vectorizer, get_model_save_name(basename='tfidf-vectorizer', suffix='joblib'))
    X_test = tf_idf_vectorizer.transform(X_test)

    if IS_SENTIMENT_READABILITY_ON:
        sr_train, sr_test, sr_y_train, sr_y_test = train_test_split(rspct_df[['sentiment_val', 'readability_score']], rspct_df['subreddit'],
                                                                    test_size=TEST_FRACTION, random_state=42)
        X_train = scipy.sparse.hstack((X_train, sr_train[['sentiment_val', 'readability_score']])).tocsr()
        X_test = scipy.sparse.hstack((X_test, sr_test[['sentiment_val', 'readability_score']])).tocsr()

    return (X_train, y_train, X_test, y_test)
