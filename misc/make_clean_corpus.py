from nltk import sent_tokenize
fin = open("../corps_II_preproc_cleaned.csv", "r", encoding='utf-8-sig')
fapp = open("../applause_only.en", "a+")
fnapp = open("../nonapplause_only.en", "a+")

for line in fin:
    toks = line.strip().split(",")
    label = toks[1]
    if label == '0':
        label = "nonapplause"
    else:
        label = "applause"
    para = toks[0]
    para = sent_tokenize(para)
    for sent in para:
        entry = label + " " + sent + "\n"
        if label == "applause":
            fapp.write(entry)
        else:
            fnapp.write(entry)

fapp.close()
fnapp.close()



