train_file = open('dev.out', 'r')
new_file = open('dev.in','w')
for line in train_file:
    try:
        word, tag = line.split()
        new_file.write(word+'\n')
    except:
        new_file.write('\n')



