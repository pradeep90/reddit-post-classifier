"""
sources:
https://evolution.ai//blog/page/5/an-imagenet-like-text-classification-task-based-on-reddit-posts/
https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
"""
import time, datetime
import pandas as pd
import numpy as np

PATH_TO_DATA = './preprocessing/'
DATA_FILE_NAME = 'rspct_preprocessed_lemmatized.tsv'
DATA = PATH_TO_DATA+DATA_FILE_NAME

df = pd.read_csv(DATA, sep='\t') 
#df = pd.read_csv(DATA, sep='\t', nrows=10000) 

train_data_percentage = 100
train_split_index = int(len(df) * (train_data_percentage/100))
df = df[:train_split_index]

# concatenate the title and selftext columns
df['text'] = df['title'] + ' ' + df['selftext']
df = df.drop(['title','selftext'], axis=1)

import lda
from sklearn.feature_extraction.text import CountVectorizer

t_start = time.time()
t0 = time.time()
df_new = pd.DataFrame(columns=['subreddit', 'text'])
i=0
print('Combining the self texts...')
for subreddit in df['subreddit'].unique():
	#tmp_df = df[df['subreddit'] == subreddit].groupby(['subreddit'])['text'].transform(lambda x: ' '.join(x)).reset_index()
	tmp_df = df[df['subreddit'] == subreddit].applymap(str).groupby(['subreddit'])['text'].transform(lambda x: ' '.join(x)).reset_index()
	combined_texts = tmp_df.iloc[0]['text']
	#print(subreddit)
	#print(combined_texts, type(combined_texts))
	df_new.loc[i] = [subreddit, combined_texts]
	i+=1
	#input('......')

print(df_new.describe())
print(df_new.head(7))
#print(df_new['subreddit'].unique())
#print(np.where(df_new['subreddit'].unique() == 'teenmom'))
#print('Combining the self texts took time: {}'.format(time.time()-t0))
#exit(0)

n_topics = 100 # number of topics
n_iter = 500 # number of iterations
#n_iter = 100 # number of iterations

# vectorizer: ignore English stopwords & words that occur less than 5 times
t0 = time.time()
NUM_FEATURES = 30000
cvectorizer = CountVectorizer(min_df=5, stop_words='english', max_features=NUM_FEATURES)
cvz = cvectorizer.fit_transform(df_new['text'])

print('cvz:')
print(cvz)
print(cvz.shape)
print('Vectorizing took time: {}'.format(time.time()-t0))

# train an LDA model
to = time.time()
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(cvz)

print(X_topics)
print('LDA took time: {}'.format(time.time()-t0))

from sklearn.manifold import TSNE

# a t-SNE model
# angle value close to 1 means sacrificing accuracy for speed
# pca initializtion usually leads to better results
t0 = time.time()
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

# 20-D -> 2-D
tsne_lda = tsne_model.fit_transform(X_topics)

print(tsne_lda)
print('TSNE took time: {}'.format(time.time()-t0))

import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool

n_top_words = 5 # number of keywords we show

# 20 colors
colormap_org = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])
colormap = np.array(['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A',
       '#D62728', '#FF9896', '#9467BD', '#C5B0D5', '#8C564B', '#C49C94',
       '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D',
       '#17BECF', '#9EDAE5'])

colormap = np.array(['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A', '#D62728', '#FF9896', '#9467BD', '#C5B0D5', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5', '#84F8CF', '#9BF4B7', '#6F4790', '#473080', '#4B9E32', '#25A9F1', '#33B5DE', '#A168F4', '#E2851F', '#072FCC', '#00FCAA', '#7CA620', '#61717A', '#48E52E', '#29A3FA', '#379A95', '#3FAA68', '#93E32E', '#C5A27B', '#945E60', '#5F1085', '#F3232D', '#424C13', '#29C88D', '#786ED6', '#8CE6FC', '#B62AA6', '#3BF9AB', '#617C08', '#8A3B70', '#BE57AA', '#DA1F33', '#4A7017', '#250D3F', '#603DC8', '#2EBD3B', '#120B63', '#5E3FF5', '#6B1F0B', '#D93385', '#237124', '#9AB3DF', '#5C1FEF', '#1433C8', '#6685B7', '#F05668', '#1D5152', '#AF803C', '#E25906', '#F1D19F', '#B6C680', '#4E06EA', '#28AB17', '#8F457A', '#F6B493', '#B7439E', '#C6D429', '#0062AB', '#517A72', '#E5C1D4', '#10CDD6', '#1754E4', '#208450', '#E4F900', '#13FDA6', '#9FEF19', '#D4602A', '#4207CD', '#D5A101', '#6D0701', '#32613C', '#659A8F', '#5D33F3', '#CB290B', '#8CE73B', '#8344B1', '#3A4F8E', '#091514', '#6984A1', '#BB15FD'])


sub_reddits = list(df_new['subreddit'].unique())
print('sub_reddits = {}'.format(sub_reddits))
_lda_keys = []
assigned_subreddits = []
for i in range(X_topics.shape[0]):
	_lda_keys +=  X_topics[i].argmax(),
	assigned_subreddits.append(sub_reddits[_lda_keys[-1]])

print('_lda_keys = {}; len(_lda_keys) = {}'.format(_lda_keys, len(_lda_keys)))
print('assigned_subreddits = {}; len(assigned_subreddits) = {}'.format(assigned_subreddits, len(assigned_subreddits)))

topic_summaries = []
topic_word = lda_model.topic_word_  # all topic words
#print('topics_word = {}'.format(topic_word))
vocab = cvectorizer.get_feature_names()
#print('vocab = {}'.format(vocab))
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
	topic_summaries.append(' '.join(topic_words)) # append!
print('topic_summaries = {},  len(topic_summaries = {}'.format(topic_summaries, len(topic_summaries)))

title = 'TSNE on Subreddit Self Posts with {} topics'.format(n_topics)
num_example = len(X_topics)

plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

#source = bp.ColumnDataSource(data=dict(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=colormap[_lda_keys][:num_example], content= df_new['text'][:num_example][:200] , topic_key= _lda_keys[:num_example]))
source = bp.ColumnDataSource(data=dict(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=colormap[_lda_keys][:num_example], content= df_new['text'][:num_example][:200] , topic_key= assigned_subreddits[:num_example]))

plot_lda.scatter(x='x', y='y',
                 color='color',
		 source=source)

hover = plot_lda.select(dict(type=HoverTool))
#hover.tooltips = [("content", "@content") , ("topic", "@topic_key")]
hover.tooltips = [("topic", "@topic_key")]

# file saving format: tsne_lda_plot_{%age of data}_{num_iters}_{num_topics}_{time.now()}.png
file_name = 'tsne_lda_plot_{}_{}_{}_{}.html'.format(train_data_percentage, n_iter, n_topics, str(datetime.datetime.now()))
save(plot_lda, file_name)

print('Total time taken: {}'.format(time.time() - t_start))
