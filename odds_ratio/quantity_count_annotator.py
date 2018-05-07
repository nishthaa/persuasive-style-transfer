class quantity_count_annotator:
    def process(self,lobject):
        if 'pos_tags' not in lobject.annotations:
            return lobject
        count = 0
        for token,tag in lobject.annotations['pos_tags']:
            if tag == 'CD':
                count = count + 1
        lobject.annotations['quantity_count'] = count
        return lobject
