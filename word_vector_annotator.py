import numpy as np

class word_vector_annotator:
    def __init__(self):
        print('Loading word vectors...')
        wordvector_file = open('/home/nimadaan/lazada/python/data/bulk_statistics/wiki.simple.vec','rb')
        firstLine = True
        self.wordvectors = {}
        for line in wordvector_file:
            if firstLine:
                firstLine = False
                continue
            tokens = line.split()

            word = tokens[0].strip()
            word = str(word.decode("utf-8"))

            vector_list = [float(val) for val in tokens[1:]]
            if len(vector_list) != 300:
                continue

            vector = np.asarray(vector_list)
            self.wordvectors[word] = vector
        print('...Word vectors loaded.')

    def process(self,lobject):
        if 'tokens' not in lobject.annotations:
            return lobject
        title_tokens = lobject.annotations['tokens']
        wordvector_dict = {}
        for token in title_tokens:
            if token in self.wordvectors:
                wordvector_dict[token] = self.wordvectors[token]
        lobject.annotations['wordvectors'] = wordvector_dict
        return lobject

