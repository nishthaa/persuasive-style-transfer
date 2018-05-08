import nltk

class pos_tag_annotator:
    def process(self,lobject):
        title = lobject.text
        tokens = title.split()
        tags = nltk.pos_tag(tokens)
        lobject.annotations["pos_tags"] = tags
        return lobject
