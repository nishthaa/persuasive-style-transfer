from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities

def tokenize_and_stem(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	stems = [stemmer.stem(t) for t in filtered_tokens]
	return stems


def tokenize_only(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

def strip_proppers(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
	return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def strip_proppers_POS(text):
	tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
	non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
	return non_propernouns


DIR_LINK = "data/convotev1.1/data_stage_one/training_set/"

titles = os.listdir(DIR_LINK)[:100]

fps = [open(DIR_LINK + file) for file in titles]

speeches = [fp.read() for fp in fps]

stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in speeches:
	allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
	totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
	
	allwords_tokenized = tokenize_only(i)
	totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
								 min_df=0.2, stop_words='english',
								 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


tfidf_matrix = tfidf_vectorizer.fit_transform(speeches)

terms = tfidf_vectorizer.get_feature_names()



dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 5

km = KMeans(n_clusters=num_clusters, random_state=10)

km.fit(tfidf_matrix)

cluster_labels = km.fit_predict(tfidf_matrix)

clusters = km.labels_.tolist()

silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
# joblib.dump(km,  'doc_cluster.pkl')
print("Silhoutte score (avg.) for n_clusters = " + str(num_clusters) + " is: " + str(silhouette_avg))

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()


films = { 'title': titles, 'speeches': speeches, 'cluster': clusters}

frame = pd.DataFrame(films, index = [clusters] , columns = ['title', 'cluster'])

frame['cluster'].value_counts() #number of films per cluster 


# print("Top terms per cluster:")
# print()
# #sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

# for i in range(num_clusters):
# 	print("Cluster %d words:" % i, end='')
	
# 	for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
# 		print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
# 	print() #add whitespace
# 	print() #add whitespace
	
# 	print("Cluster %d titles:" % i, end='')
# 	for title in frame.ix[i]['title'].values.tolist():
# 		print(' %s,' % title, end='')
# 	print() #add whitespace
# 	print() #add whitespace
	
# print()
# print()

preprocess = [strip_proppers(doc) for doc in speeches]
tokenized_text = [tokenize_and_stem(text) for text in preprocess]
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]


#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus, num_topics=5, 
							id2word=dictionary, 
							update_every=5, 
							chunksize=10000, 
							passes=100)

lda.show_topics()

topics_matrix = lda.show_topics(formatted=False, num_words=20)
#print(topics_matrix)
#topics_matrix = np.array(topics_matrix)

# topic_words = topics_matrix[:,:,1]
# for i in topic_words:
# 	print([str(word) for word in i])
# 	print()

for topic in topics_matrix:

	words = topic[1]

	chars = []

	for word in words:

		chars.append(word[0])

	print(chars)


