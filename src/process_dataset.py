import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from parameters import IS_DEBUGGING_ON

# TODO(pradeep): Extract this.
RUNNING_KAGGLE_KERNEL = True

def join_text(row):
    if RUNNING_KAGGLE_KERNEL:
        return row['title'][:100] + " " + row['selftext'][:512]
    else:
        return row['title'] + " " + row['selftext']

def get_reddit_dataset(dataset_name='data/rspct.tsv', size=100):
    rspct_df = pd.read_csv(dataset_name, sep='\t')

    # DATASET_SIZE = 100
    # DATASET_SIZE = 100000

    if IS_DEBUGGING_ON:
        print('Dataset size:', size)

    if size is not None:
        rspct_df = rspct_df.head(size)
    return rspct_df

def get_vectorized_training_and_test_set():
    """Return vectorized, label-encoded training and test set with labels."""

    # print(os.listdir("../input"))

    # running our benchmark code in this kernel lead to memory errors, so
    # we do a slightly less memory intensive procedure if this is True,
    # set this as False if you are running on a computer with a lot of RAM
    # it should be possible to use less memory in this kernel using generators
    # rather than storing everything in RAM, but we won't explore that here
    rspct_df = get_reddit_dataset()

    # we join the title and selftext into one field

    rspct_df['text'] = rspct_df[['title', 'selftext']].apply(join_text, axis=1)

    train_split_index = int(len(rspct_df) * 0.8)

    X_train, X_test, y_train, y_test = train_test_split(rspct_df['text'], rspct_df['subreddit'],
                                                         test_size=0.2, random_state=42)

    # label encode y

    le = LabelEncoder()
    # le.fit(y_train)
    le.fit(pd.concat([y_train, y_test]))

    # TODO(pradeep): Change this back.
    # old_y_train = y_train.copy()

    y_train = le.transform(y_train)
    y_test  = le.transform(y_test)

    # print(y_train[:5])

    # array([920, 931, 161, 827, 669])

    # extract features from text using bag-of-words (single words + bigrams)
    # use tfidf weighting (helps a little for Naive Bayes in general)
    # note : you can do better than this by extracting more features, then
    # doing feature selection, but not enough memory on this kernel!

    # print('this cell will take about 10 minutes to run')

    NUM_FEATURES = 30000 if RUNNING_KAGGLE_KERNEL else 100000

    # TODO(pradeep): Use max_df and stop_words='english'. Try analyzer = 'word' and 'char'.
    tf_idf_vectorizer = TfidfVectorizer(max_features = NUM_FEATURES,
                                    min_df=5,
                                    ngram_range=(1,2),
                                    stop_words=None,
                                    token_pattern='(?u)\\b\\w+\\b',
                                )

    X_train = tf_idf_vectorizer.fit_transform(X_train)
    X_test  = tf_idf_vectorizer.transform(X_test)

    # if we have more memory, select top 100000 features and select good features
    if not RUNNING_KAGGLE_KERNEL:
        # TODO(pradeep): This doesn't reduce the number of features because there
        # are already only NUM_FEATURES of them.
        chi2_selector = SelectKBest(chi2, NUM_FEATURES)

        chi2_selector.fit(X_train, y_train)

        X_train = chi2_selector.transform(X_train)
        X_test  = chi2_selector.transform(X_test)

    # print(X_train.shape, X_test.shape)

    # this cell will take about 10 minutes to run

    # ((810400, 30000), (202600, 30000))

    return (X_train, y_train, X_test, y_test)
