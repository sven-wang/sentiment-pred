train_file = open('dev.out', 'r')
new_file = open('iob.test','w')
for line in train_file:
    try:
        word, tag = line.split()
        new_file.write(word+' dummy '+tag[0] + '\n')
    except:
        new_file.write('\n')

train_file = open('train', 'r')
new_file = open('iob.train','w')
for line in train_file:
    try:
        word, tag = line.split()
        new_file.write(word+' dummy '+tag[0] + '\n')
    except:
        new_file.write('\n')



