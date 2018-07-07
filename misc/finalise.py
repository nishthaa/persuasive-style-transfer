import sys
from random import shuffle

FILE = sys.argv[1]
FILE_WRITE = sys.argv[2]

fh = open(FILE)

elems = []

for line in fh:
	line = line.strip()
	elems.append(line)


shuffle(elems)

fw = open(FILE_WRITE,"a+")

for i in range(len(elems)):
	fw.write(elems[i]+"\n")

	

