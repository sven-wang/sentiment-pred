# for quick access, to get the location of each sentiment in the dictionary
dic = {'START': 0, 'B': 1, 'I': 2, 'O': 3, 'STOP': 4}
l = ['START', 'B', 'I', 'O', 'STOP']

import pprint

pp = pprint.PrettyPrinter(indent=4)


def train():
    ############################# initialize parameter ####################################

    # store emission parameters
    # data structure: tuple + dictionary
    e_count = ({}, {}, {}, {})  ## 1st dict empty (start no emission)
    e_param = ({}, {}, {}, {})  ## 1st dict empty (start no emission)

    # store transition parameters
    # initialize as a 5*4 matrix of zeros
    w, h = 5, 4
    t_param = [[0] * w for i in range(h)]

    # count # of sentiment appears+1
    """data structure: vector [4] (from START to 'O')
          vector[i] = sum(t_param[i])+1
    """
    count = [0] * 4

    ## read and parse file
    train_file = open('train_EN', 'r')
    u = 'START'
    obs_space = set()

    for obs in train_file:
        try:
            obs, v = obs.split()
            obs = obs.strip().lower()
            v = v.strip()
            v = v[0]  ## get BIO
            position = dic[v]  ## position: 1~7

            # preprocessing of some special cases:
            if obs[:7] == 'http://':
                obs = 'http://'

            # update e_count
            if (obs in e_count[position]):
                e_count[position][obs] += 1
            else:
                e_count[position][obs] = 1

            # update t_param
            pre_position = dic[u]
            t_param[pre_position][position] += 1
            u = v

            # add into train_obs_set
            if obs not in obs_space:
                obs_space.add(obs)

        except:
            # meaning the end of a sentence: x->STOP
            pre_position = dic[u]
            t_param[pre_position][4] += 1
            u = 'START'

    # get count(yi)+1
    for i in range(0, 4):
        temp_sum = 0
        for j in range(0, 5):
            temp_sum = temp_sum + t_param[i][j]
        count[i] = temp_sum + 1
    # print (t_param)
    # print (count)
    # print (e_count)

    ## convert transision param to probablity
    for i in range(0, 4):
        for j in range(0, 4):
            t_param[i][j] = 1.0 * t_param[i][j] / count[i]
    # print (t_param)

    # building emission params table: a list of 4 dicts, each dict has all obs as keys,
    # value is 0 if obs never appears for this state

    for i in range(1, 4):  # state 1-7
        for obs in obs_space:
            if obs not in e_count[i]:
                e_param[i][obs] = 0.01 / max(
                    count)  ## ???????????????????????????????????????????????????????????????????????????????????????? whether should be 0?? or lowest prob of all
            else:
                e_param[i][obs] = 1.0 * e_count[i][obs] / count[i]

    return obs_space, e_param, t_param, count


################################# Naive Bayes Sentiment Analysis Train ############################
dic_sentiment = {'negative': 0, 'neutral': 1, 'positive': 2}
l_sentiment = ['negative', 'neutral', 'positive']
not_set = set(['not', 'but', 'yet', 'any', 'no'])  ## for negation
meaningless_set = set(['i', 'is', 'are', 'do', 'does', 'did', ',', 'a', '.', 'am', 'an', 'because'])  ## eliminate words


def sentimentTrain():
    ############### Initialize Parameter #################

    ## store the sentiment for each word appear in the training set
    dic_words = {}

    ## read and parse file
    train_file = open('train_EN', 'r')
    not_flag = False
    sentiment_score = 0  # initialize neutral sentiment
    X = []

    for obs in train_file:
        try:
            obs, v = obs.split()
            obs = obs.strip().lower()

            if obs[:7]=='http://':
                obs = 'http://'

            v = v.strip()
            # collect all the words in sentence
            if (obs not in meaningless_set):
                X.append(obs)
            # calculate the sentiment of the sentence: negative: -1 netrual: 0 positive: +1
            if (v[0] == 'B'):
                sentiment_score += dic_sentiment[v.split('-')[1]] - 1
        except:
            # meaning the end of a sentence
            for w in X:
                # assumption: any word's sentiment that appears after negation is the opposite of sentence sentiment
                # if (w in not_set):
                #   not_flag = not not_flag
                if (not_flag):
                    sentiment_score *= -1
                # fill up the dic
                # get sentiment position first:
                if (sentiment_score == 0):
                    p = 1
                else:
                    p = (sentiment_score > 0) * 2  # if true: p = 2, if false: p = 0
                if (w in dic_words):
                    dic_words[w][p] += 1
                else:
                    dic_words[w] = [0, 0, 0]
                    dic_words[w][p] += 1
            X = []
            sentiment_score = 0
            not_flag = False

    # print(dic_words)
    return dic_words


############################### Sentiment Training ######################################


def getGlobalPosNegWords():
    posSet = set()
    negSet = set()
    pos_file = open("wordSent/" + "positive-words.txt", 'r')
    neg_file = open("wordSent/" + "negative-words.txt", 'r')

    for line in pos_file:
        line = line.strip()
        if line:
            if line[0] != ';':
                posSet.add(line)

    for line in neg_file:
        line = line.strip()
        if line:
            if line[0] != ';':
                negSet.add(line)
    # print (posSet)
    # print (negSet)

    return posSet, negSet

def getEffectivePosNegWords():
    dic_sentiment = {'negative': 0, 'neutral': 1, 'positive': 2}
    posSet, negSet = getGlobalPosNegWords()
    train_file = open('train_EN', 'r')
    # not_flag = False
    sentiment_score = 0  # initialize neutral sentiment
    X = []

    effectivePosDict = {}
    effectiveNegDict = {}

    for obs in train_file:
        try:
            obs, v = obs.split()
            obs = obs.strip().lower()
            v = v.strip()
            # collect all the words in sentence
            X.append(obs)
            # calculate the sentiment of the sentence: negative: -1 netrual: 0 positive: +1
            if (v[0] == 'B'):
                sentiment_score += dic_sentiment[v.split('-')[1]] - 1
        except:
            # meaning the end of a sentence
            if (sentiment_score > 0):
                for w in X:
                    if w in posSet:
                        if w in effectivePosDict:
                            effectivePosDict[w] += 1
                        else:
                            effectivePosDict[w] = 1

            elif (sentiment_score < 0):
                for w in X:
                    if w in negSet:
                        if w in effectiveNegDict:
                            effectiveNegDict[w] += 1
                        else:
                            effectiveNegDict[w] = 1
            X = []
            sentiment_score = 0
            # not_flag = False
    return effectivePosDict, effectiveNegDict

############################## Decoding ######################################


def forward(preScore, x):
    """inputs: preScore: list of (pre_score real_num, pre_parent int)
               x: current word
       output: list of max(score real_num, parent int) for all states, len=7
    """
    layer = []
    for i in range(1, 4):  # i: 1~7
        temp_score = []
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, 4):  # j:1-7
            # score = preScore*a*b
            j_score = preScore[j - 1][0] * (t_param[j][i]) * b  # trans 1~7 -> 1-7
            temp_score.append(j_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value)  # index: 0-6
        layer.append((max_value, max_index))
    return layer


def viterbiAlgo(X, dic_words, effPosSet, effNegSet):
    """input: X, words list
       output: Y, sentiment list
    """
    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # start -> 1
    x = X[0]
    for j in range(1, 4):
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        prev_layer.append((prob, 0))  # (prob, START)
    layers = [[(1, -1)], prev_layer]

    # calculate path i=(1,...,n)
    for i in range(1, n):  # 1 -> n-1
        score = forward(layers[i], X[i])  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)

    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(1, 4):
        # score = preScore*a
        t_score = layers[n][j - 1][0] * (t_param[j][4])
        temp_score.append(t_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value)
    layers.append([(max_value, max_index)])
    # print(scores)

    # backtracking
    parent = 0  # only 1 entry in STOP
    for i in range(n + 1, 1, -1):  # index range from N to 2
        parent = layers[i][parent][1]
        Y.insert(0, l[parent + 1])  # 1-7
    # print(Y)

    ########calculate sentiment
    not_flag = False
    sentiment_score = [0.0, 0.0, 0.0]

    for w in X:
        w = w.lower()
        # if (w in not_set):
        #   not_flag = not not_flag

        if (w in dic_words):
            if sum(dic_words[w]) < 3:
                continue
            elif sum(dic_words[w]) > 50:
                continue

            if w in effPosSet:
                sentiment_score[2] += effPosSet[w]
            if w in effNegSet:
                sentiment_score[0] += effNegSet[w]

            for i in range(0, 3):
                sentiment_score[i] += dic_words[w][i]  # /(dic_words[w][0]+dic_words[w][1]+dic_words[w][2])

    max_sentiment = max(sentiment_score)
    sentiment = l_sentiment[sentiment_score.index(max_sentiment)]
    # sentiment = "neutral"
    for i in range(0, len(Y)):
        if (Y[i] != 'O'):
            Y[i] += '-' + sentiment
    # print(Y)
    return Y


def runPart3(obs_space, e_param, t_param, count, dic_words):
    dev_file = open('dev_EN.in', 'r')
    out_file = open('dev_EN.p5_noNegation.out', 'w')

    effPosSet, effNegSet = getEffectivePosNegWords()
    X = []
    for r in dev_file:
        r = r.strip().lower()

        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X, dic_words, effPosSet, effNegSet)
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            # preprocessing of some special cases:
            if r[:7] == 'http://':
                r = 'http://'
            X.append(r)


obs_space, e_param, t_param, count = train()
dic_words = sentimentTrain()

pp.pprint(dic_words)
runPart3(obs_space, e_param, t_param, count, dic_words)
print("finish!")