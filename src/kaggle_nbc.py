import numpy as np
import pandas as pd
import os

# print(os.listdir("../input"))

# running our benchmark code in this kernel lead to memory errors, so
# we do a slightly less memory intensive procedure if this is True,
# set this as False if you are running on a computer with a lot of RAM
# it should be possible to use less memory in this kernel using generators
# rather than storing everything in RAM, but we won't explore that here
RUNNING_KAGGLE_KERNEL = True

rspct_df = pd.read_csv('data/rspct.tsv', sep='\t')

info_df  = pd.read_csv('data/subreddit_info.csv')

# Basic data analysis

# print(rspct_df.head(5))

# 	id 	subreddit 	title 	selftext
# 0 	6d8knd 	talesfromtechsupport 	Remember your command line switches... 	Hi there, <lb>The usual. Long time lerker, fi...
# 1 	58mbft 	teenmom 	So what was Matt "addicted" to? 	Did he ever say what his addiction was or is h...
# 2 	8f73s7 	Harley 	No Club Colors 	Funny story. I went to college in Las Vegas. T...
# 3 	6ti6re 	ringdoorbell 	Not door bell, but floodlight mount height. 	I know this is a sub for the 'Ring Doorbell' b...
# 4 	77sxto 	intel 	Worried about my 8700k small fft/data stress r... 	Prime95 (regardless of version) and OCCT both,...

# note that info_df has information on subreddits that are not in data,
# we filter them out here

info_df = info_df[info_df.in_data].reset_index()
# print(info_df.head(5))

# 	index 	subreddit 	category_1 	category_2 	category_3 	in_data 	reason_for_exclusion
# 0 	0 	whatsthatbook 	advice/question 	book 	NaN 	True 	NaN
# 1 	25 	theydidthemath 	advice/question 	calculations 	NaN 	True 	NaN
# 2 	26 	datarecovery 	advice/question 	data recovery 	NaN 	True 	NaN
# 3 	27 	declutter 	advice/question 	declutter 	NaN 	True 	NaN
# 4 	30 	productivity 	advice/question 	discipline 	NaN 	True 	NaN
# Naive Bayes benchmark

# we join the title and selftext into one field

def join_text(row):
    if RUNNING_KAGGLE_KERNEL:
        return row['title'][:100] + " " + row['selftext'][:512]
    else:
        return row['title'] + " " + row['selftext']

rspct_df['text'] = rspct_df[['title', 'selftext']].apply(join_text, axis=1)

# take the last 20% as a test set - N.B data is already randomly shuffled,
# and last 20% is a stratified split (equal proportions of subreddits)

DATASET_SIZE = 1000
# DATASET_SIZE = 100000

rspct_df = rspct_df.head(DATASET_SIZE)

train_split_index = int(len(rspct_df) * 0.8)

# TODO(pradeep): Use `train_test_split`. Save files.
train_df, test_df = rspct_df[:train_split_index], rspct_df[train_split_index:]
X_train , X_test  = train_df.text, test_df.text
y_train, y_test   = train_df.subreddit, test_df.subreddit

from sklearn.preprocessing import LabelEncoder

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

from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.feature_selection import chi2, SelectKBest

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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# train a naive bayes model, get predictions

def get_model(name='NBC'):
    model = None
    if name == 'NBC':
        model = MultinomialNB(alpha=0.1)
    elif name == 'LR':
        model = LogisticRegression(solver='sag')
    return model

# model_name = 'NBC'
model_name = 'LR'
model = get_model(model_name)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# we use precision-at-k metrics to evaluate performance
# (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

def precision_at_k(y_true, y_pred, k=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.argsort(y_pred, axis=1)
    y_pred = y_pred[:, ::-1][:, :k]
    arr = [y in s for y, s in zip(y_true, y_pred)]
    return np.mean(arr)

# print(f'{y_test[:10]}')
# print(f'{y_pred[:10]}')

# Got DeprecationWarning.
# print('precision@1 =', np.mean(y_test == y_pred))
print('precision@1 =', precision_at_k(y_test, y_pred_proba, 1))
print('precision@3 =', precision_at_k(y_test, y_pred_proba, 3))
print('precision@5 =', precision_at_k(y_test, y_pred_proba, 5))

# RUNNING_KAGGLE_KERNEL == True
# precision@1 = 0.610528134254689
# precision@3 = 0.7573692003948668
# precision@5 = 0.8067670286278381

# RUNNING_KAGGLE_KERNEL == False
# precision@1 = 0.7292102665350444
# precision@3 = 0.8512240868706812
# precision@5 = 0.8861500493583415

# precision@1 = 0.615187561697927
# precision@3 = 0.7615399802566634
# precision@5 = 0.8105972359328727
