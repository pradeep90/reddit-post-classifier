import os, sys
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import preprocessing


class TestPreprocessing(unittest.TestCase):

	post_to_test = 'Hi there,  <lb>The usual. Long time lerker, first time poster, be kind etc. Sorry if this isn\'t the right place...<lb><lb>Alright. Here\'s the story. I\'m an independent developer who produces my own software. We\'re going to call me well, $me.<lb><lb>I work with $dev who helps to produce software with me. We use $PopularVersionControl.<lb><lb>We\'re trying to remove a branch that was created by mistake.  The branch is beta1. We want just beta.<lb><lb>&gt; $me: "$dev, can you rename that branch because we\'re going to use just two. I don\'t want to keep up with 80 quintilian branches."  <lb>&gt; $dev: "sure, one second."<lb><lb>Five minutes later...<lb><lb>&gt; $dev: "[CurseWords] I want beta1 to die!"  <lb>&gt; $me: "What happened?"<lb><lb>Lots of removed dialog where $dev explains what he did...<lb><lb>&gt; $me: "Did you try $PopularVersionControl with -u?"  <lb>&gt; $dev: "[Cursing] That would be why!"<lb><lb>In short. Always check your command line switches...They are important!<lb>'
	data = [[post_to_test]]
	df = pd.DataFrame(data, columns= ['selfpost'])
	

	def test_lower_remove_tags_puncts_whites(self):
		"""
		Tests if lower-casing and removing punctuations, and tags
		"""
		expected_res = 'hi there  the usual long time lerker first time poster be kind etc sorry if this isnt the right placealright heres the story im an independent developer who produces my own software were going to call me well mei work with dev who helps to produce software with me we use popularversioncontrolwere trying to remove a branch that was created by mistake  the branch is beta1 we want just beta me dev can you rename that branch because were going to use just two i dont want to keep up with 80 quintilian branches   dev sure one secondfive minutes later dev  i want beta1 to die   me what happenedlots of removed dialog where dev explains what he did me did you try popularversioncontrol with u   dev  that would be whyin short always check your command line switchesthey are important'
		data = [[expected_res]]
		expected_df = pd.DataFrame(data, columns= ['selfpost'])

		res = preprocessing.preprocess(self.df)
		both_equal = res.equals(expected_df)

		self.assertEqual(both_equal, True)


	def test_stemming(self):
		expected_res = 'hi there the usual long time lerker first time poster be kind etc sorri if thi isnt the right placealright here the stori im an independ develop who produc my own softwar were go to call me well mei work with dev who help to produc softwar with me we use popularversioncontrolwer tri to remov a branch that wa creat by mistak the branch is beta1 we want just beta me dev can you renam that branch becaus were go to use just two i dont want to keep up with 80 quintilian branch dev sure one secondf minut later dev i want beta1 to die me what happenedlot of remov dialog where dev explain what he did me did you tri popularversioncontrol with u dev that would be whyin short alway check your command line switchesthey are import' 

		data = [[expected_res]]
		expected_df = pd.DataFrame(data, columns= ['selfpost'])

		res = preprocessing.preprocess(self.df)
		both_equal = res.equals(expected_df)

		self.assertEqual(both_equal, True)


	def test_lemmatization(self):
		expected_res = 'hi there the usual long time lerker first time poster be kind etc sorry if this isnt the right placealright here the story im an independent developer who produce my own software were going to call me well mei work with dev who help to produce software with me we use popularversioncontrolwere trying to remove a branch that wa created by mistake the branch is beta1 we want just beta me dev can you rename that branch because were going to use just two i dont want to keep up with 80 quintilian branch dev sure one secondfive minute later dev i want beta1 to die me what happenedlots of removed dialog where dev explains what he did me did you try popularversioncontrol with u dev that would be whyin short always check your command line switchesthey are important'

		data = [[expected_res]]
		expected_df = pd.DataFrame(data, columns= ['selfpost'])

		res = preprocessing.preprocess(self.df)
		both_equal = res.equals(expected_df)

		self.assertEqual(both_equal, True)
		

	def test_preprocessing_whole(self):
		"""
		Tests the whole preprocessing pipelines and compares if the final result is as expected.
		"""
		expected_res = ''
		data = [[expected_res]]
		expected_df = pd.DataFrame(data, columns= ['selfpost'])

		res = preprocessing.preprocess(self.df)
		both_equal = res.equals(expected_df)

		self.assertEqual(both_equal, True)

if __name__ == '__main__':
	unittest.main()
