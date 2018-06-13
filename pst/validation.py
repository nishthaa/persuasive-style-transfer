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
import sys
import codecs
from sklearn import feature_extraction
import mpld3
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from odds_ratio import get_odds_ratio
from nltk import word_tokenize


def pre_process(text, stop_words):

    lower = text.lower()
    mappings = {}
    content = lower.split()
    word_list = []
    for i in content:
        if(('@' not in i) and ('<.*?>' not in i) and i.isalnum() and (not i in stop_words)):

            word_list += [i]

    number_tokens = [re.sub(r'[\d]', ' ', i) for i in word_list]
    number_tokens = ' '.join(number_tokens).split()
    stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
    for i in range(len(stemmed_tokens)):
        if stemmed_tokens[i] not in mappings and len(stemmed_tokens[i]) > 1:

            mappings[stemmed_tokens[i]] = word_list[i]

    length_tokens = [i for i in stemmed_tokens if len(i) > 1]
    return length_tokens, mappings


DIR_LINK_TRAIN = "data/convotev1.1/data_stage_one/training_set/"
DIR_LINK_TEST = "data/convotev1.1/data_stage_one/test_set/"


titles = os.listdir(DIR_LINK_TRAIN)

tmp = []
for speech in titles:

    if speech[-5] == "Y":
        tmp.append(speech)
titles = tmp

titles_test = os.listdir(DIR_LINK_TEST)

tmp_test = []
tmp_test_2 = []
for speech in titles_test:
    tmp_test.append(speech)
    if speech[-5] == "Y":
        tmp_test_2.append(speech)

titles_test = tmp_test
titles_test_2 = tmp_test_2

fps = [open(DIR_LINK_TRAIN + file) for file in titles]

speeches = [fp.read() for fp in fps]

for fp in fps:

    fp.close()

fps_test = [open(DIR_LINK_TEST + file) for file in titles_test]

speeches_test = [fp.read() for fp in fps_test]

for fp in fps_test:

    fp.close()

fps_test_2 = [open(DIR_LINK_TEST + file) for file in titles_test_2]

speeches_test_2 = [fp.read() for fp in fps_test_2]

for fp in fps_test_2:

    fp.close()


labelled_sentence = gensim.models.doc2vec.TaggedDocument

all_speeches = []
all_speeches_w2v = []
texts = []
j = 0
k = 0

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(nltk.corpus.stopwords.words('english'))
p_stemmer = PorterStemmer()

mappings = None
for speech in speeches:

    clean_speech, mappings = pre_process(speech, stop_words)

    if clean_speech:

        all_speeches.append(labelled_sentence(clean_speech, [j]))

        j += 1

    k += 1

for speech in speeches_test:

    speech = word_tokenize(speech)

    all_speeches_w2v.append(speech)


print("Number of emails processed: ", k)
print("Number of non-empty emails vectors: ", j)


d2v_model = Doc2Vec(all_speeches, vector_size=2000, window=10, min_count=500, workers=7, dm=1,
                    alpha=0.025, min_alpha=0.001)

d2v_model.train(all_speeches, total_examples=d2v_model.corpus_count,
                epochs=10, start_alpha=0.002, end_alpha=-0.016)

# print(all_speeches_w2v)

model = Word2Vec(all_speeches_w2v)
# print(model)


# print(d2v_model.docvecs.most_similar(1))


test_speeches = os.listdir(DIR_LINK_TEST)
tmp = []
for speech in test_speeches:

    if speech[-5] == "Y":
        tmp.append(speech)
test_speeches = tmp


tfps = [open(DIR_LINK_TEST + file) for file in test_speeches]

tspeeches = [fp.read() for fp in tfps]

vectors = []

for content in tspeeches:

    tmp, mp = pre_process(content, stop_words)
    vectors.append(d2v_model.infer_vector(tmp))


num_clusters = 10

km = KMeans(n_clusters=num_clusters, random_state=10)


# km.fit(vectors)

cluster_labels = km.fit_predict(vectors)

clusters = km.labels_.tolist()

print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid


# print(clusters)

silhouette_avg = silhouette_score(vectors, cluster_labels)
# joblib.dump(km,  'doc_cluster.pkl')
print("Silhoutte score (avg.) for n_clusters = " +
      str(num_clusters) + " is: " + str(silhouette_avg))

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()

tracker = {}

for speech in test_speeches:

    if speech[:3] not in tracker:
        tracker[speech[:3]] = [0] * num_clusters

for i in range(len(test_speeches)):

    tracker[test_speeches[i][:3]][clusters[i]] += 1

print(tracker)

highORwords, lowORwords = get_odds_ratio(os.listdir(DIR_LINK_TEST))

replacement = {}


for word1 in highORwords:

    max_sim = -1

    for word2 in lowORwords:

        sim = model.similarity(word1, word2)

        if sim > max_sim:

            max_sim = sim
            replacement[word1] = word2

print(replacement)

final_speeches = []

count_replaced = 0

for speech in speeches_test_2:

    tmp = speech.split(" ")

    for i in range(len(tmp)):

        if tmp[i] in replacement:
            count_replaced += 1
            # print(count_replaced)
            tmp[i] = replacement[tmp[i]]

    final_speeches.append(" ".join(tmp))


vectors_final = []

for content in final_speeches:

    vectors_final.append(d2v_model.infer_vector(
        pre_process(content, stop_words)))

num_clusters = 10

km2 = KMeans(n_clusters=num_clusters, random_state=10)

# km.fit(vectors)

cluster_labels2 = km2.fit_predict(vectors_final)

clusters2 = km2.labels_.tolist()

# print(clusters)

silhouette_avg2 = silhouette_score(vectors_final, cluster_labels2)
# joblib.dump(km,  'doc_cluster.pkl')
print("Silhoutte score (avg.) for n_clusters = " +
      str(num_clusters) + " is (after replacement): " + str(silhouette_avg2))

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()

tracker2 = {}

for speech in titles_test_2:

    if speech[:3] not in tracker2:
        tracker2[speech[:3]] = [0] * num_clusters

for i in range(len(titles_test_2)):

    tracker2[titles_test_2[i][:3]][clusters2[i]] += 1

print(tracker2)
