from nltk import word_tokenize

class title_tokenizer:
    def process(self,lobject):
        _title = lobject.text.lower()
        tokens = []
        try:
            tokens = word_tokenize(_title)
        except:
            tokens = _title.split()
            pass
        lobject.annotations['tokens'] = tokens
        return lobject

