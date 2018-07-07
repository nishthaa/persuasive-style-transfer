fin = open("../processed_pers_data.txt", "r")

pers_count = 0
non_pers_count = 0

for line in fin:
    toks = line.strip().split()
    if toks[0] == "persuasive":
        pers_count+=1
    else:
        non_pers_count+=1

print(pers_count)
print(non_pers_count)


