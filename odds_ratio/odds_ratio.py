from nltk import word_tokenize
from title_tokenizer import title_tokenizer
from pos_tag_annotator import pos_tag_annotator
from quantity_count_annotator import quantity_count_annotator
from stopword_annotator import stopword_annotator
from word_vector_annotator import word_vector_annotator
import os


def process(lobject):
	_title = lobject.lower()
	tokens = []
	try:
		tokens = word_tokenize(_title)
	except:
		tokens = _title.split()
		pass
	lobject = tokens
	return lobject

#Read the file


def get_odds_ratio(reader):
	title_tokenizer_instance = title_tokenizer()
	num_pos_comments = 0
	num_neg_comments = 0
	word_num_pos_comments = {}
	word_num_neg_comments = {}
	all_word_set = set()

	for file in reader:
		try:
			fp = open("/Volumes/Brihi/convote_v1.1/data_stage_one/test_set/"+file)
			text = fp.read()
			tag = file[-5]

			#Annotator Pipeline
			toks = process(text)

			#Doing something with annotator output
			all_word_set.update(toks)
			seen_tokens = []
			if tag == 'Y':
				num_pos_comments = num_pos_comments + 1
				for token in toks:
					if token in seen_tokens:
						continue
					else:
						seen_tokens.append(token)
					if token in word_num_pos_comments:
						word_num_pos_comments[token] = word_num_pos_comments[token] + 1
					else:
						word_num_pos_comments[token] = 1

			if tag == 'N':
				num_neg_comments = num_neg_comments + 1
				for token in toks:
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


reader = os.listdir("/Volumes/Brihi/convote_v1.1/data_stage_one/test_set/")
get_odds_ratio(reader)
	
	


