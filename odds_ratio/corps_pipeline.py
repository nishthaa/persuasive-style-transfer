import json
from cmv_object import cmv_object
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

#Import Annotators
from title_tokenizer import title_tokenizer
from pos_tag_annotator import pos_tag_annotator
from quantity_count_annotator import quantity_count_annotator
from stopword_annotator import stopword_annotator
from word_vector_annotator import word_vector_annotator


#
def variance(list):
    highORWordVectors = []
    mean = np.zeros(300)
    count = 0
    for word in list:
        if word not in word_vector_annotator_instance.wordvectors:
            continue
        vector = word_vector_annotator_instance.wordvectors[word]
        highORWordVectors.append(vector)
        mean = mean + vector
        count = count + 1
    mean = mean / (count*1.0)

    variance = 0
    for vector in highORWordVectors:
        variance = variance + np.linalg.norm(vector - mean) ** 2.0

    variance = variance / (count*1.0)
    variance = math.sqrt(variance)
    print(variance)

def eigenvalues(list):
    highORWordVectors = []
    mean = np.zeros(300)
    count = 0
    for word in list:
        if word not in word_vector_annotator_instance.wordvectors:
            continue
        vector = word_vector_annotator_instance.wordvectors[word]
        highORWordVectors.append(vector)
        mean = mean + vector
        count = count + 1
    mean = mean / (count*1.0)

    centered_vectors = []
    for vector in highORWordVectors:
        centered_vectors.append((vector - mean).tolist())

    X = np.matrix(centered_vectors)
    C = np.dot(np.transpose(X),X)
    # eigv, eigw = np.linalg.eig(C)
    U,s,V = np.linalg.svd(X)
    print(s)
    plt.plot(s)
    # print np.real(eigv)
    # plt.plot(np.real(eigv))


#Read the file
trainingfile = open('/home/nimadaan/cmv/pythonwksp/data/corps_full_preproc.csv','rb')
reader = csv.reader(x.replace('\0', '') for x in trainingfile)

#Annotator Initializations
title_tokenizer_instance = title_tokenizer()
pos_tag_annotator_instance = pos_tag_annotator()
quantity_count_annotator_instance = quantity_count_annotator()
stopword_annotator_instance = stopword_annotator()
word_vector_annotator_instance = word_vector_annotator()


num_pos_comments = 0
num_neg_comments = 0
word_num_pos_comments = {}
word_num_neg_comments = {}
all_word_set = set()

progress_count = 0
for row in reader:
    try:
        progress_count = progress_count + 1
        if progress_count%100 == 0:
            print("Progress Count : ",progress_count)
        text = row[0]
        tag = row[1]

        #Get data
        data = cmv_object(text)
        data.label = tag

        #Annotator Pipeline
        data = title_tokenizer_instance.process(data)
        # data = pos_tag_annotator_instance.process(data)
        # data = quantity_count_annotator_instance.process(data)
        # data = stopword_annotator_instance.process(data)
        # data = word_vector_annotator_instance.process(data)

        #Doing something with annotator output
        all_word_set.update(data.annotations['tokens'])
        seen_tokens = []
        if tag == '1':
            num_pos_comments = num_pos_comments + 1
            for token in data.annotations['tokens']:
                if token in seen_tokens:
                    continue
                else:
                    seen_tokens.append(token)
                if token in word_num_pos_comments:
                    word_num_pos_comments[token] = word_num_pos_comments[token] + 1
                else:
                    word_num_pos_comments[token] = 1

        if tag == '0':
            num_neg_comments = num_neg_comments + 1
            for token in data.annotations['tokens']:
                if token in seen_tokens:
                    continue
                else:
                    seen_tokens.append(token)
                if token in word_num_neg_comments:
                    word_num_neg_comments[token] = word_num_neg_comments[token] + 1
                else:
                    word_num_neg_comments[token] = 1
    except:
        pass


#Computing Odds Ratio for each word
oddsratiodict = {}
word_comment_counts = {}
for word in all_word_set:
    pwp = 0
    nwp = 0
    if word in word_num_pos_comments:
        pwp = word_num_pos_comments[word]
    if word in word_num_neg_comments:
        nwp = word_num_neg_comments[word]
    pwa = num_pos_comments - pwp
    nwa = num_neg_comments - nwp
    word_comment_counts[word] = (pwp,pwa,nwp,nwa)
    if pwp == 0 or pwa == 0 or nwp == 0 or nwa == 0:
        continue
    oddsratio = (pwp*nwa*1.0)/(pwa*nwp*1.0)
    oddsratiodict[word] = oddsratio

#Sort map by descending odds ratio
limit = 100
count = 0
highORWords = []
print('High Odds Ratio:')
for word in sorted(oddsratiodict, key=oddsratiodict.get, reverse=True):
    if count >= limit:
        break
    pwp, pwa, nwp, nwa = word_comment_counts[word]
    if pwp <= 1 or nwp <= 1:
        continue
    print(word, str(oddsratiodict[word]), str(word_comment_counts[word]))
    highORWords.append(word)
    count = count + 1


print('\n\nLow Odds Ratio:')
count = 0
lowORWords = []
for word in sorted(oddsratiodict, key=oddsratiodict.get):
    if count >= limit:
        break
    pwp, pwa, nwp, nwa = word_comment_counts[word]
    if pwp <= 1 or nwp <= 1:
        continue
    print(word, str(oddsratiodict[word]), str(word_comment_counts[word]))
    lowORWords.append(word)
    count = count + 1

# variance(highORWords)
# variance(lowORWords)
# eigenvalues(highORWords)
# eigenvalues(lowORWords)
# plt.show()



