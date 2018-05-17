import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from odds_ratio import get_odds_ratio
import string
import sys
from gensim.models import Word2Vec
from nltk import word_tokenize
import operator


stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in
# the text that it is passed


def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes

    :param clusters: the cluster assignments array
    :type clusters: numpy.array

    :param classes: the ground truth classes
    :type classes: numpy.array

    :returns: the purity score
    :rtype: float

    ref: http://www.caner.io/purity-in-python.html
    """

    A = np.c_[(clusters, classes)]

    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is
    # caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw
    # punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is
    # caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw
    # punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


DIR_LINK_TRAIN = "data/convotev1.1/data_stage_one/training_set/"
DIR_LINK_TEST = "data/convotev1.1/data_stage_one/test_set/"
THRESHOLD = int(sys.argv[1])
PUNCTUATION = string.punctuation
print(THRESHOLD)
training_speeches = os.listdir(DIR_LINK_TRAIN)


tmp = []
tmpn = []
label_ref = {}
labels = []
z = 0
for speech in training_speeches:

    if speech[-5] == "Y":
        tmp.append(speech)
        if speech[:3] not in label_ref:
            label_ref[speech[:3]] = z
            z += 1
        labels.append(label_ref[speech[:3]])
    if speech[-5] == "N":
        tmpn.append(speech)

# Speeches from the training set labelled "Y"
training_speeches = tmp
training_speeches_n = tmpn


fps = [open(DIR_LINK_TRAIN + file) for file in training_speeches]
fps_n = [open(DIR_LINK_TRAIN + file) for file in training_speeches_n]

# Content of the training speeches labelled "Y"
train_content = [fp.read() for fp in fps]
train_content_n = [fp.read() for fp in fps_n]

for fp in fps:

    fp.close()

for fp in fps_n:

    fp.close()

totalvocab_stemmed = []
totalvocab_tokenized = []

for i in train_content:
    # for each item in 'synopses', tokenize/stem
    allwords_stemmed = tokenize_and_stem(i)
    # extend the 'totalvocab_stemmed' list
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame(
    {'words': totalvocab_tokenized}, index=totalvocab_stemmed)


# print(vocab_frame.head())


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))


tfidf_matrix = tfidf_vectorizer.fit_transform(
    train_content)  # fit the vectorizer to synopses


# print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

# dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 38

km = KMeans(n_clusters=num_clusters, random_state=10)

km.fit(tfidf_matrix)


clusters = km.labels_.tolist()

print("Purity score: ", purity_score(clusters, labels))


speeches = {'title': training_speeches,
            'speech_content': train_content, 'cluster': clusters}


frame = pd.DataFrame(speeches, index=[clusters], columns=[
                     'title', 'cluster'])

print(frame['cluster'].value_counts())


print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, :: -1]

central_words = []

true_k = np.unique(labels).shape[0]

terms = tfidf_vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='\n')
    print()

print()
print()

# print(central_words)

tracker = {}

for speech in training_speeches:

    if speech[:3] not in tracker:
        tracker[speech[:3]] = [0] * num_clusters

for i in range(len(training_speeches)):

    tracker[training_speeches[i][:3]][clusters[i]] += 1

for key in tracker:

    mx = -1

    for i in tracker[key]:

        if i > mx:

            mx = i

    print(key, i)

highORwords, lowORwords = get_odds_ratio(os.listdir(DIR_LINK_TRAIN))

yes_words = {}
no_words = {}

for speech in train_content:

    speech = speech.strip(PUNCTUATION)
    toks = speech.split(" ")
    toks = [tok.strip() for tok in toks]
    for i in range(len(toks)):
        if toks[i] not in yes_words:
            yes_words[toks[i]] = 1
        else:
            yes_words[toks[i]] += 1

for speech in train_content_n:

    speech = " ".join([word.strip(PUNCTUATION) for word in speech.split(" ")])
    toks = speech.split(" ")
    toks = [tok.strip() for tok in toks]
    for i in range(len(toks)):
        if toks[i] not in no_words:
            no_words[toks[i]] = 1
        else:
            no_words[toks[i]] += 1

above_threshold = []

for key in yes_words.keys():
    if yes_words[key] > THRESHOLD and key in no_words:
        above_threshold.append((key, yes_words[key]))

above_threshold = sorted(above_threshold, key=operator.itemgetter(1))
all_speeches = train_content + train_content_n
all_speeches_w2v = []


for speech in all_speeches:

    speech = word_tokenize(speech)
    all_speeches_w2v.append(speech)

all_speeches.append(["unsound"])


model = Word2Vec(all_speeches_w2v)

mapping = {}
i = 0

for tup in above_threshold:
    word = tup[0]
    max_sim = -1
    if i < 100:
        mapping[word] = ""
        i += 1
    else:
        for wrd in lowORwords:
            wrd = wrd.strip()
            word = word.strip()
            try:
                if model.similarity(word, wrd) > max_sim:
                    mapping[word] = wrd
            except:
                print(tup, wrd)

final_speeches = []

count_replaced = 0

for speech in train_content:

    tmp = speech.split(" ")

    for i in range(len(tmp)):

        if tmp[i] in mapping:
            count_replaced += 1
            # print(count_replaced)
            tmp[i] = mapping[tmp[i]]

    final_speeches.append(" ".join(tmp))


# Now working on replaces speeches

print("After replacement")


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))


tfidf_matrix = tfidf_vectorizer.fit_transform(
    final_speeches)  # fit the vectorizer to synopses


terms = tfidf_vectorizer.get_feature_names()

# dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 38

km = KMeans(n_clusters=num_clusters, random_state=10)

km.fit(tfidf_matrix)


clusters_aft = km.labels_.tolist()

print("Purity score: ", purity_score(clusters_aft, labels))


speeches = {'title': training_speeches,
            'speech_content': final_speeches, 'cluster': clusters_aft}


frame = pd.DataFrame(speeches, index=[clusters_aft], columns=[
                     'title', 'cluster'])

print(frame['cluster'].value_counts())


print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, :: -1]

central_words = []

true_k = np.unique(labels).shape[0]

terms = tfidf_vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='\n')
    print()

print()
print()

# Tracker for the replaced speeches -------------

tracker = {}

for speech in final_speeches:

    if speech[:3] not in tracker:
        tracker[speech[:3]] = [0] * num_clusters

for i in range(len(final_speeches)):

    tracker[final_speeches[i][:3]][clusters_aft[i]] += 1

for key in tracker:

    mx = -1

    for i in tracker[key]:

        if i > mx:

            mx = i

    print(key, i)
