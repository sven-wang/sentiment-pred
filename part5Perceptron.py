import numpy as np
import unicodedata
import sys
############################# initialize parameter ####################################
dic = {'PRESTART': 0, 'START': 1, 'B-positive': 2, 'I-positive': 3, 'B-neutral': 4,
       'I-neutral': 5,'B-negative': 6,'I-negative': 7,'O': 8, 'STOP': 9, 'POSTSTOP': 10}
l = ['PRESTART', 'START', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative',
     'I-negative', 'O', 'STOP', 'POSTSTOP']

# store emission parameters
# data structure: tuple + dictionary
e_param = ({}, {}, {}, {}, {}, {}, {}, {}, {})  ## 1st,2nd dict empty
obs_space = set()

# store transition parameters
# initialize as a 11*11*11 matrix of zeros
a = 11
t_param = np.zeros((11, 11, 11))

# b_inSpace = float(sys.argv[1])
# b_notInSpace = float(sys.argv[2])  # 1/count+1
b_inSpace = 0
b_notInSpace = 0
# num of iteration over the training set
T = 20


def preprocess(word):
    word = word.lower()
    word = word.decode('utf-8')
    word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
    if 'http://' in word or 'https://' in word:
        word = 'http://'
    # elif 'xc3' in word:
    #     word = "xcsChar"
    elif 'www.' in word:
        word = 'www.'
    #
    elif any(char.isdigit() for char in word):
        word = '***numbers***'
    # no use
    # elif '?' in word:
    #     word = '??'
    # elif '!' in word:
    #     word = '!!'
    return word

def forward(preScore, x):
    """inputs: preScore: list of (pre_score real_num, pre_parent int)
               x: current word
       output: list of max(score real_num, parent int) for all states, len=7
    """

    # k j i
    layer = []
    for i in range(2, 9):  # i: 2~8
        temp_score = []
        # calculate emission first
        if ((x in obs_space) & (x in e_param[i])):
            b = e_param[i][x]
        elif (x in obs_space):
            b = b_inSpace                                       # !!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            b = b_notInSpace
        for j in range(2, 9):  #
            # score = preScore*a*b
            for k in range(2, 9):
                kj_score = preScore[j - 2][0] + t_param[k][j][i] + b  # trans 2~8 -> 2-8
                temp_score.append(kj_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value) / 7
        layer.append((max_value, max_index + 2)) # index: 0-6 + 2
    return layer


def viterbiAlgo(X):
    # initialization
    n = len(X)
    Y = []
    prev_layer = []

    # prestart, start -> 1
    x = X[0]
    for j in range(2, 9): # 2-8
        if ((x in obs_space) & (x in e_param[j])):
            b = e_param[j][x]
        elif (x in obs_space):
            b = b_inSpace                                                        # to be tuned
        else:
            b = b_notInSpace
        prob = t_param[0][1][j] + b   # prestart * start * y1
        prev_layer.append((prob, 1))  # (prob, PRESTART, START)            ????
    layers = [[(1, -1)], prev_layer]


    # start, 1 -> 2
    if len(X) > 1:
        x = X[1]
        layer = []
        for j in range(2, 9): # 2-8
            temp_score = []
            if ((x in obs_space) & (x in e_param[j])):
                b = e_param[j][x]
            elif (x in obs_space):
                b = b_inSpace                                                           # to be tuned
            else:
                b = b_notInSpace
            for k in range(2,9):
                temp_score.append(t_param[1][k][j] + b)   #  start + y1 + y2
            max_value = max(temp_score)
            max_index = temp_score.index(max_value)
            layer.append((max_value, max_index + 2))
        layers.append(layer)  #


    # calculate path i=(2,...,n)

    for i in range(2, n):  # 2 -> n-1
        score = forward(layers[i], X[i])  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)


    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(2, 9):
        for k in range(2,9):
            # score = preScore*a
            kj_score = layers[n][j-2][0] + (t_param[k][j][dic['STOP']])
            temp_score.append(kj_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value) / 7
    layers.append([(max_value, max_index + 2)])
    # print(scores)

    # backtracking
    parent = 2  # only 1 entry in STOP
    for i in range(n + 1, 1, -1):  # index range from N to 2
        parent = layers[i][parent-2][1] # 0-6
        Y.insert(0, l[parent])  # 2-8
    # print(Y)
    return Y


def updateParam(X, Y, Ytrain):
    for i in range(2, len(Y)):
        t_param[dic[Y[i - 2]]][dic[Y[i - 1]]][dic[Y[i]]] += 1
        t_param[dic[Ytrain[i - 2]]][dic[Ytrain[i - 1]]][dic[Ytrain[i]]] -= 1
    for i in range(2, len(Y) - 2):
        if (X[i - 2] in e_param[dic[Y[i]]]):
            e_param[dic[Y[i]]][X[i - 2]] += 1
        elif (X[i - 2] in obs_space):
            e_param[dic[Y[i]]][X[i - 2]] = 1
        else:
            e_param[dic[Y[i]]][X[i - 2]] = 1
            obs_space.add(X[i - 2])

    for i in range(2, len(Y) - 2):
        if (X[i - 2] in e_param[dic[Ytrain[i]]]):
            e_param[dic[Ytrain[i]]][X[i - 2]] -= 1
        elif (X[i - 2] in obs_space):
            e_param[dic[Ytrain[i]]][X[i - 2]] = -1
        else:
            e_param[dic[Ytrain[i]]][X[i - 2]] = -1
            obs_space.add(X[i - 2])



def train(type):
    ############################# initialize parameter ####################################

    for trainStep in range(T):
        print ('Iteration: ', trainStep)
        ## read and parse file
        train_file = open(type+'/train', 'r')
        Ygold = ['PRESTART', 'START']
        X = []

        for obs in train_file:
            try:
                obs, v = obs.split()
                obs = obs.strip()
                obs = preprocess(obs)
                v = v.strip()
                X.append(obs)
                Ygold.append(v)
            except:
                # meaning the end of a sentence: x->STOP
                Ygold.extend(['STOP', 'POSTSTOP'])
                Ytrain = ['PRESTART', 'START']
                Ytrain.extend(viterbiAlgo(X))
                Ytrain.extend(['STOP', 'POSTSTOP'])

                updateParam(X, Ygold, Ytrain)

                # reset
                Ygold = ['PRESTART', 'START']
                X = []


def runPerceptron(type):
    dev_file = open(type+'/dev.in', 'r')
    out_file = open(type+'/dev.perceptron.out', 'w')
    # dev_file = open(type+'/test.in', 'r')
    # out_file = open(type+'/test.out', 'w')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X)
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            r = preprocess(r)
            X.append(r)

#
# updateParam(['New', 'Year', ',', 'New'],
#             ['PRESTART', 'START', 'B-positive', 'B-positive', 'O', 'O', 'STOP', 'POSTSTOP'],
#             ['PRESTART', 'START', 'O', 'O', 'O', 'O', 'STOP', 'POSTSTOP'])
# viterbiAlgo(['New', 'Year',','])

# for type in ['ES_test','EN_test']:
for type in ['ES', 'EN']:
    train(type)
    runPerceptron(type)
# print(e_param)
# print(obs_space)
# print(t_param)