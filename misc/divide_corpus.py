from sklearn.cross_validation import train_test_split
import sys

FILE = sys.argv[1]

fh = open(FILE)

X_file = []
y_file = []

for line in fh:
	line = line.strip().split(" ")
	label = line[0]
	line = line[1:]
	line = " ".join(line)
	y_file.append(label)
	X_file.append(line)

X_txt, X, y_txt, y = train_test_split(X_file,y_file, test_size = 0.75, random_state=1)

X_ct, X_dtxt, y_ct, y_dtxt = train_test_split(X_txt, y_txt, test_size=0.025, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=1)



ftrain = open(".."+"".join(FILE.split(".")[-2])+".train.en","a+")
ftest = open(".."+"".join(FILE.split(".")[-2])+".test.en","a+")
fdev = open(".."+"".join(FILE.split(".")[-2])+".dev.en","a+")

fctxt = open(".."+"".join(FILE.split(".")[-2])+"_classtrain.txt","a+")
fdtxt = open(".."+"".join(FILE.split(".")[-2])+"_dev.txt","a+")


for i in range(len(X_train)):
	ftrain.write(y_train[i]+" "+X_train[i]+"\n")

for i in range(len(X_dev)):
	fdev.write(y_dev[i]+" "+X_dev[i]+"\n")

for i in range(len(X_test)):
	ftest.write(X_test[i]+"\n")

for i in range(len(X_ct)):
	fctxt.write(y_ct[i]+" "+X_ct[i]+"\n")

for i in range(len(X_dtxt)):
	fdtxt.write(y_dtxt[i]+" "+X_dtxt[i]+"\n")