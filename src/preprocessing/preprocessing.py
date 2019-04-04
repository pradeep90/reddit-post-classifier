# Source: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908

import re 

import pandas as pd
import numpy as np

DEBUG		= True

PATH_TO_DATA		= '../../data/'
DATA_FILE_NAME		= 'rspct.tsv' 
DATA				= PATH_TO_DATA+DATA_FILE_NAME 

OUTPUT_FILE_NAME 	= 'rspct_preprocessed.tsv'


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
	# to remove &gt tags. TODO: might be other such tags that need to be removed
	p1 = re.compile(r'&gt')
	text = p1.sub('', text)

	# to remove tags inside {}, [] and HTML tags
	p2 = re.compile(r'[<{\[].*?[>}\]]') 
	text = p2.sub('', text)

	# to remove punctuations (after removing tags etc.)
	puncts_to_remove = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
	text = text.translate({ord(c): None for c in puncts_to_remove})

	return text.strip()


def remove_tags_puncts_whitespaces(data):
	"""
	Removes punctuations, HTML tags, and other tags inside {} or [] brackets and whitespaces
	"""
	if DEBUG:
		print('Removing punctuations, tags and whitespaces')
	for col in data.columns:
		if col not in ['id', 'subreddit']:
			data[col] = data[col].apply(remove_tags_puncts_whites)
	return data


def preprocess(data):
	data = lowercase(data) 
	#data = remove_nums(data)
	data = remove_tags_puncts_whitespaces(data)

	# DISCUSS: tokenization and stop-word-removal to be done by the library itself?

	# TODO: stemming, lemmatization

	return data


def main():
	if DEBUG:
		print('Reading the data')

	df = pd.read_csv(DATA, sep='\t')
	
	preprocessed_df = preprocess(df)

	preprocessed_df.to_csv(OUTPUT_FILE_NAME, sep='\t', index=False)


if __name__ == '__main__':
	main()