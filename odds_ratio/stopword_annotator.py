from nltk.corpus import stopwords

class stopword_annotator:
    def __init__(self):
        self.stopword_set = set(stopwords.words('english'))

    def process(self,lobject):
        _title = lobject.text.lower()
        if 'tokens' not in lobject.annotations:
            return lobject
        stopword_list = []
        for token in lobject.annotations['tokens']:
            if token in self.stopword_set:
                stopword_list.append(token.lower())
        lobject.annotations['stopwords'] = stopword_list
        return lobject

