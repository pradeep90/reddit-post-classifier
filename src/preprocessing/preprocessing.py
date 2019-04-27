# Source: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908

# reading level of posts coming out as negative for most posts because of the way they are written

import re 

import pandas as pd
import numpy as np

import time

import nltk
nltk.download('wordnet') #TODO: should add this to the make file?

# for sentiment analysis
from textblob import TextBlob
# for readability score
import textstat

DEBUG	= True
TIME_DEBUG = True
INCLUDE_SENTIMENT_FEATURE = True
INCLUDE_READABILITY_FEATURE = True

PATH_TO_DATA 	= '../../data/'
DATA_FILE_NAME 	= 'rspct.tsv' 
DATA 		= PATH_TO_DATA+DATA_FILE_NAME 

OUTPUT_FILE_NAME	= 'rspct_preprocessed_sentiment_readability_stemmed.tsv'


def lowercase(df):
	if DEBUG:
		print('Lowercasing the dataset')
	df=df.apply(lambda x: x.astype(str).str.lower())
	return df


def remove_nums(df):
	if DEBUG:
		print('Removing numbers from all attributes except id')
	for col in df.columns:
		if col not in ['id']:
			df[col] = df[col].str.replace('\d+', '')
	return df


def remove_tags_puncts_whites(text):
	text = text.strip()
	
	# to remove &gt tags. TODO: there might be other such tags that need to be removed
	p1 = re.compile(r'&gt|&amp|&lt')
	text = p1.sub(' ', text)

	# to remove tags inside {}, [] and HTML tags
	p2 = re.compile(r'[<{\[].*?[>}\]]') 
	text = p2.sub(' ', text)

	# remove single quotes only if they preceded or follow a word
	text1 = re.sub(r"((?P<a>\s)'(?P<x>\w))|((?P<y>\w)'(?P<b>\s))|((?P<c>\s)'(?P<d>\s))", r'\g<a>\g<x>\g<y>\g<b>\g<c>\g<d>', text)
	text = re.sub(r"((?P<a>\s)'(?P<x>\w))|((?P<y>\w)'(?P<b>\s))|((?P<c>\s)'(?P<d>\s))", r'\g<a>\g<x>\g<y>\g<b>\g<c>\g<d>', text1)
	text = text.strip("'")

	# to remove punctuations (after removing tags etc.)
	#puncts_to_remove = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
	puncts_to_remove = """!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
	text = text.translate({ord(c): ' ' for c in puncts_to_remove})

	# print('ret:', text.strip())
	return text.strip()

def remove_tags_puncts_whitespaces(data):
	"""
	Removes punctuations, HTML tags, and other tags inside {} or [] brackets and whitespaces
	"""
	if DEBUG:
		print('Removing punctuations, tags and whitespaces')
	for col in data.columns:
		if col not in ['id', 'subreddit', 'sentiment_val']:
			data[col] = data[col].apply(remove_tags_puncts_whites)
	return data


def stem_text(text):
	stemmer = nltk.stem.PorterStemmer()
	tokenized_text = nltk.tokenize.word_tokenize(text)
	stemmed_words = [stemmer.stem(word) for word in tokenized_text]
	#print('stemmed=',' '.join(stemmed_words))
	return ' '.join(stemmed_words)


def stem_data(data):
	"""
	Replace all words with their stem words
	"""
	if DEBUG:
		print ('Stemming the data')
	for col in data.columns:
		if col not in ['id', 'subreddit', 'sentiment_val']:
			data[col] = data[col].apply(stem_text)
	return data	


def lemmatize_text(text):
	lemmatizer = nltk.stem.WordNetLemmatizer()
	tokenized_text=nltk.tokenize.word_tokenize(text)
	lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_text]
	#print('lemmatized:',' '.join(lemmatized_words))
	return ' '.join(lemmatized_words)


def lemmatize_data(data):
	if DEBUG:
		print ('Lemmatizing the data')
	for col in data.columns:
		if col not in ['id', 'subreddit', 'sentiment_val']:
			data[col] = data[col].apply(lemmatize_text)
	return data		


def text_sentiment(text):
    """
    Given input text, returns a scalar estimate of the sentiment of that text.
    Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement.
    """
    # return indicoio.sentiment_hq(text)
    return TextBlob(text).sentiment.polarity


def include_sentiment(data):
	data['title_selftext'] = data['title'] + ' ' + data['selftext']
	data['sentiment_val'] = data['title_selftext'].apply(text_sentiment)
	data = data.drop(['title_selftext'], 1)
	return data


def text_readability_score(text):
	return textstat.flesch_reading_ease(text)


def include_readability_score(data):
	data['title_selftext'] = data['title'] + ' ' + data['selftext']
	data['readability_score'] = data['title_selftext'].apply(text_readability_score)
	data = data.drop(['readability_score'], 1)
	return data


def preprocess(data):

	t0 = time.time()
	data = lowercase(data) 	
	t1 = time.time()
	if TIME_DEBUG:
		print('Lowercasing took time: {}'.format(t1-t0))

	t0 = time.time()
	if INCLUDE_SENTIMENT_FEATURE:
		data = include_sentiment(data)
		if TIME_DEBUG:
			print('Sentiment calculations took time: {}'.format(time.time()-t0))

	t0 = time.time()
	if INCLUDE_READABILITY_FEATURE:
		data = include_readability_score(data)
		if TIME_DEBUG:
			print('Readability score calculations took time: {}'.format(time.time()-t0))

	# t0 = time.time()
	# data = remove_nums(data)
	# t1 = time.time()
	# if TIME_DEBUG:
	# 	print('That took time: {}'.format(t1-t0))

	t0 = time.time()
	data = remove_tags_puncts_whitespaces(data)
	t1 = time.time()
	if TIME_DEBUG:
		print('Removing punctuations took time: {}'.format(t1-t0))

	# """
	t0 = time.time()
	data = stem_data(data)
	t1 = time.time()
	if DEBUG:
		print('Stemming took time: {}'.format(t1-t0))
	# """

	"""
	t0 = time.time()
	data = lemmatize_data(data)
	t1 = time.time()
	if DEBUG:
		print('Lemmatization took time: {}'.format(t1-t0))
	"""

	return data


def main():
	if DEBUG:
		print('Reading the data')

	t0 = time.time()

	df = pd.read_csv(DATA, sep='\t')
	# df = pd.read_csv(DATA, sep='\t', nrows=1000)
	
	preprocessed_df = preprocess(df)

	preprocessed_df.to_csv(OUTPUT_FILE_NAME, sep='\t', index=False)

	t1 = time.time()
	if TIME_DEBUG:
		print('Total time taken: {}'.format(t1-t0))

if __name__ == '__main__':
	main()
