import pprint
pp = pprint.PrettyPrinter(indent=5)

# for quick access, to get the location of each sentiment in the dictionary
dic = {'START': 0, 'B-positive': 1, 'B-neutral': 2, 'B-negative': 3, 'I-positive': 4, 'I-neutral': 5, 'I-negative': 6,
       'O': 7, 'STOP': 8}
l = ['START', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative', 'O', 'STOP']

def train(type):
    ############################# initialize parameter ####################################

    # store emission parameters
    # data structure: tuple + dictionary
    e_count = ({}, {}, {}, {}, {}, {}, {}, {})  ## 1st dict empty (start no emission)
    e_param = ({}, {}, {}, {}, {}, {}, {}, {})  ## 1st dict empty (start no emission)

    # store transition parameters
    # initialize as a 9*8 matrix of zeros
    w, h = 9, 8
    t_param = [[0] * w for i in range(h)]

    # count # of sentiment appears+1
    """data structure: vector [8] (from START to 'O')
          vector[i] = sum(t_param[i])+1
    """
    count = [0] * 8

    ## read and parse file
    train_file = open(type+'/train', 'r')
    u = 'START'
    obs_space = set()

    for obs in train_file:
        try:
            obs, v = obs.split()
            obs = obs.strip()
            v = v.strip()
            position = dic[v]  ## position: 1~7
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
            t_param[pre_position][8] += 1
            u = 'START'

    # get count(yi)+1
    for i in range(0, 8):
        temp_sum = 0
        for j in range(0, 9):
            temp_sum = temp_sum + t_param[i][j]
        count[i] = temp_sum + 1
    # print (t_param)
    # pp.pprint (e_count)
    # print (count)
    # print e_count

    ## convert transision param to probablity
    for i in range(0, 8):
        for j in range(0, 8):
            t_param[i][j] = 1.0 * t_param[i][j] / count[i]
    # print (t_param)

    # building emission params table: a list of 8 dicts, each dict has all obs as keys,
    # value is 0 if obs never appears for this state

    for i in range(1,8): # state 1-7
        for obs in obs_space:
            if obs not in e_count[i]:
                e_param[i][obs] = 0.5 / max(count) ## ???????????????????????????????????????????????????????????????????????????????????????? whether should be 0?? or lowest prob of all
            else:
                e_param[i][obs] = 1.0 * e_count[i][obs] / count[i]

    return obs_space, e_param, t_param, count


############################################## PART 2 ###############################################
def runPart2(type,obs_space, e_param, count):
    dev_file = open(type+'/dev.in','r')
    output_file = open(type+'/dev.p2.out','w')
    for o in dev_file:
        o = o.strip()
        if (o== ''):
            output_file.write('\n')
            continue
        temp_list = []
        #y* = argmax(e(x|y))
        for j in range(1,8):
            if (o in obs_space):
                temp_list.append(e_param[j][o])
            else:
                temp_list.append(1.0/count[j])
        max_value = max(temp_list)
        max_index = temp_list.index(max_value)   # 0-6
        output_file.write(o + " " + l[max_index+1] + '\n') # 1-7


############################### Part 3 ######################################

def forward(preScore, x):
    """inputs: preScore: list of (pre_score real_num, pre_parent int)
               x: current word
       output: list of max(score real_num, parent int) for all states, len=7
    """
    layer = []
    for i in range(1, 8):  # i: 1~7
        temp_score = []
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, 8):  # j:1-7
            # score = preScore*a*b
            j_score = preScore[j-1][0] * (t_param[j][i]) * b  # trans 1~7 -> 1-7
            temp_score.append(j_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value)  # index: 0-6
        layer.append((max_value, max_index))
    return layer


def viterbiAlgo(X):
    """input: X, words list
       output: Y, sentiment list
    """
    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # start -> 1
    x = X[0]
    for j in range(1, 8):
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        prev_layer.append((prob, 0))  # (prob, START)
    layers = [[(1,-1)],prev_layer]

    # calculate path i=(1,...,n)
    for i in range(1, n):  # 1 -> n-1
        score = forward(layers[i], X[i])  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)

    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(1, 8):
        # score = preScore*a
        t_score = layers[n][j-1][0] * (t_param[j][8])
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
    print(Y)
    return Y


def runPart3(type,obs_space, e_param, t_param, count):
    dev_file = open(type+'/dev.in', 'r')
    out_file = open(type+'/dev.p3.out', 'w')
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
            X.append(r)


# for type in ["EN", "CN", "SG", "ES"]:
for type in ["EN"]:
    obs_space, e_param, t_param, count = train(type)
    runPart2(type,obs_space, e_param, count)
    runPart3(type,obs_space, e_param, t_param, count)

    # pp.pprint(count)
