import pprint
pp = pprint.PrettyPrinter(indent=5)

# for quick access, to get the location of each sentiment in the dictionary
dic = {'START': 0, 'B': 1, 'I': 2, 'O': 3, 'STOP': 4}
l = ['START', 'B', 'I', 'O', 'STOP']
negationWords = {'not', 'no'}

def train(type):
    ############################# initialize parameter ####################################

    # store emission parameters
    # data structure: tuple + dictionary
    e_count = ({}, {}, {}, {})  ## 1st dict empty (start no emission)
    e_count_back = ({}, {}, {}, {})
    e_count_fwd = ({}, {}, {}, {})
    e_param = ({}, {}, {}, {})  ## 1st dict empty (start no emission)
    e_param_back = ({}, {}, {}, {})
    e_param_fwd = ({}, {}, {}, {})

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
    train_file = open(type+'/iob.train', 'r')
    u = 'START'
    obs_space = set()

    prev_obs = "Xstart"

    for line in train_file:
        try:
            obs, v = line.split()
            obs = obs.strip().lower()
            v = v.strip()
            position = dic[v]  ## position: 1~3

            # preprocessing of some special cases:
            if obs[:7]=='http://':
                obs = 'http://'

            # update e_count
            if (obs in e_count[position]):
                e_count[position][obs] += 1
            else:
                e_count[position][obs] = 1

            # emission from current state to prev obs: position -> prev_obs
            if prev_obs != "Xstart":
                if (prev_obs in e_count_back[position]):
                    e_count_back[position][prev_obs] += 1
                else:
                    e_count_back[position][prev_obs] = 1

            # update t_param
            pre_position = dic[u]
            t_param[pre_position][position] += 1
            u = v

            # emission from prev state to current obs: pre_position -> obs
            if pre_position!=0:
                if (obs in e_count_fwd[pre_position]):
                    e_count_fwd[pre_position][obs]  += 1
                else:
                    e_count_fwd[pre_position][obs] = 1



            # add into train_obs_set
            if obs not in obs_space:
                obs_space.add(obs)

            prev_obs = obs

        except:
            # meaning the end of a sentence: x->STOP
            pre_position = dic[u]
            t_param[pre_position][4] += 1
            u = 'START'
            prev_obs = "Xstart"

    # get count(yi)+1
    for i in range(0, 4):
        temp_sum = 0
        for j in range(0, 5):
            temp_sum = temp_sum + t_param[i][j]
        count[i] = temp_sum + 1
    # print (t_param)
    # pp.pprint (e_count)
    # print (count)
    # print e_count

    ## convert transision param to probablity
    for i in range(0, 4):
        for j in range(0, 4):
            t_param[i][j] = 1.0 * t_param[i][j] / count[i]
    # print (t_param)


    # building emission params table: a list of 4 dicts, each dict has all obs as keys,
    # value is 0 if obs never appears for this state

    for i in range(1,4): # state 1-3
        for obs in obs_space:
            if obs not in e_count[i]:
                e_param[i][obs] = 0.01 / max(count) ## ???????????????????????????????????????????????????????????????????????????????????????? whether should be 0?? or lowest prob of all
            else:
                e_param[i][obs] = 1.0 * e_count[i][obs] / count[i]
            if obs not in e_count_back[i]:
                e_param_back[i][obs] = 0  # 0.01 / max(count)
            else:
                e_param_back[i][obs] = 1.0 * e_count_back[i][obs] / count[i]


            if obs not in e_count_fwd[i]:
                e_param_fwd[i][obs] = 0  # 0.01 / max(count)
            else:
                e_param_fwd[i][obs] = 1.0 * e_count_fwd[i][obs] / count[i]

        # for convenience later
        e_param_fwd[i]["Xstop"] = 1
        e_param_back[i]["Xstart"] = 1




    return obs_space, e_param, t_param, count, e_param_back, e_param_fwd


############################### Part 3 ######################################

def forward(preScore, prev_x, x, next_x):
    side_emission_weight = 0.0001
    layer = []
    for i in range(1, 4):  # i: 1~3
        temp_score = []
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        if (prev_x in obs_space):
            b_back = e_param_back[i][prev_x]
        else:
            b_back = 1.0 / count[i]
        if (next_x in obs_space):
            b_fwd = e_param_fwd[i][next_x]
        else:
            b_fwd = 1.0 / count[i]

        for j in range(1, 4):  # j:1-7
            # score = preScore*a*b
            j_score = preScore[j-1][0] * (t_param[j][i]) * (b + side_emission_weight * b_back + side_emission_weight * b_fwd)  # trans 1~7 -> 1-7
            temp_score.append(j_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value)  # index: 0-2
        layer.append((max_value, max_index))
    return layer


def viterbiAlgo(obs_sequence):
    """input: X, words list
       output: Y, sentiment list
    """
    # initialization
    n = len(obs_sequence)
    X = list(obs_sequence)
    X.append("Xstop")
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
    layers = [[(1,-1)],prev_layer]

    # calculate path i=(1,...,n)
    for i in range(1, n):  # 1 -> n-1
        score = forward(layers[i], X[i-1], X[i], X[i+1])  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)

    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(1, 4):
        # score = preScore*a
        t_score = layers[n][j-1][0] * (t_param[j][4])
        temp_score.append(t_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value)
    layers.append([(max_value, max_index)])
    # pp.pprint(scores)

    # backtracking
    parent = 0  # only 1 entry in STOP
    for i in range(n+1, 1, -1):  # index range from N to 2
        parent = layers[i][parent][1]
        Y.insert(0, l[parent + 1])  # 1-7
    # print(Y)
    return Y

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


def runPart3(type,obs_space, e_param, t_param, count):
    dev_file = open(type+'/dev.in', 'r')
    out_file = open(type+'/dev.p5.window.out', 'w')
    X = []

    effPosSet, effNegSet = getEffectivePosNegWords()
    globalPosSet, globalNegSet = getGlobalPosNegWords()
    globalWeight = 0.5


    # pp.pprint(effPosSet)
    # pp.pprint(effNegSet)


    for r in dev_file:
        r = r.strip().lower()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X)

            ### SENTIMENT ANALYTSIS
            # check polar words in X sentence
            isPos = 0
            isNeg = 0
            negation_flag = False
            for word in X:
                if word in effPosSet:
                    isPos += effPosSet[word]
                    # print ("PosWord: "+word)
                # elif word in globalPosSet:
                #     isPos += globalWeight

                if word in effNegSet:
                    isNeg += effNegSet[word]
                    # print ("NegWord: " + word)
                # elif word in globalNegSet:
                #     isNeg += globalWeight
                if word in negationWords:
                    negation_flag = True

            polarity = isPos - isNeg
            # if negation_flag:
            #     polarity = -polarity

            if polarity > 1:
                sentiment = "-positive"
                # print ('Pos: ' + ' '.join(X))
            elif polarity < -1:
                sentiment = "-negative"
                # print ('Neg: ' + ' '.join(X))
            else:
                sentiment = "-neutral"

            # hardcode sentiment to -neutral
            sentiment = "-neutral"
            for i in range(0, len(X)):
                if Y[i] == 'O':
                    out_file.write('' + X[i] + " " + Y[i] + '\n')
                else:
                    out_file.write('' + X[i] + " " + Y[i]+sentiment + '\n')
            out_file.write('\n')
            X = []
        else:
            # preprocessing of some special cases:
            if r[:7]=='http://':
                r = 'http://'
            X.append(r)


# for type in ["EN", "CN", "SG", "ES"]:
for type in [ "EN_BIO"]:
    print "Doing " + type
    obs_space, e_param, t_param, count, e_param_back, e_param_fwd = train(type)
    # runPart2(type,obs_space, e_param, count)
    runPart3(type,obs_space, e_param, t_param, count)

