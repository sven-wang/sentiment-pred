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

############################### Part 4 ######################################

# TODO: layer = [[state 1],[],...,[state 7]]
# [state 1] = [[1st (0)]],...,[kth (k-1)]]
# [1st] = [prob, parent_index (0,n), parent_sub (0,k-1)]

def forward(prev_layer, x, k):
    """inputs: prev_layer: list of list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states
               x: current word
               k: top k best
       output: list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states, len=7
    """
    layer = []
    for i in range(1, 8):  # i: 1~7
        temp_score = []
        states = []
        n = len(prev_layer[0])
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, 8):  # j:1-7
            for sub in range(0, n): # n scores for each prev_node
                # score = prev_layer*a*b
                j_score = prev_layer[j-1][sub][0] * (t_param[j][i]) * b
                temp_score.append([j_score, j-1, sub])  # 7*n scores with their parents
        temp_score.sort(key=lambda tup:tup[0],reverse=True) # sort by j_score
        for sub in range(0, k):   # get top k best
            states.append(temp_score[sub])
        layer.append(states)
    return layer


def viterbiAlgo(X, k):
    """input:  X, words list
               k, top k best
       output: Y, sentiment list
    """
    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # calculate layer (start ->) 1
    x = X[0]
    for j in range(1, 8):
        state = []
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        state.append([prob, 0, 0])  # [prob, START, 1st best]
        prev_layer.append(state)
    layers = [[(1, -1, 0)], prev_layer]
    # pp.pprint(prev_layer)


    # calculate layer (2,...,n)
    for i in range(1, n):  # prev_layer: 1 -> n-1
        layer = forward(layers[i], X[i], k)  # a list of top k best scores for all 7 states
        layers.append(layer)
        # pp.pprint("--------layer "+ str(i)+"----------")
        # pp.pprint(layer)


    # calculate layer n+1 (STOP), and get top k best
    layer = []
    temp_score = []
    states = []
    failed = False
    for j in range(1, 8):  # j:1-7
        for sub in range(0, len(layers[n][0])):  # kth score for each prev_node
            # score = prev_layer*a
            t_score = layers[n][j - 1][sub][0] * (t_param[j][8]) # TODO: for ES data set, index out of range caused by sub, for sub = 1,2,3,4
            temp_score.append([t_score, j - 1, sub])  # 7*k scores with thier parents

    temp_score.sort(key=lambda tup: tup[0], reverse=True)  # sort by j_score
    for sub in range(0, k):  # get top k best
        states.append(temp_score[sub])
    layer.append(states)
    layers.append(layer)
    # pp.pprint(layer)
    # pp.pprint(layers)

    # backtracking
    parent_index = 0    # only 1 state in STOP
    parent_sub = k-1   # kth best score in STOP layer
    for i in range(n+1, 1, -1):  # index range from N to 2
        a = layers[i][parent_index][parent_sub][1]
        b = layers[i][parent_index][parent_sub][2]
        Y.insert(0, l[a + 1])  # 1-7
        parent_index = a
        parent_sub = b
    # print(Y)
    return Y


def runPart4(type,obs_space, e_param, t_param, count, k):
    dev_file = open(type+'/dev.in', 'r')
    out_file = open(type+'/dev.p4.out', 'w')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X, k)
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)


for type in [ "ES","EN"]:
# for type in ["EN"]:

    print "Doing " + type
    obs_space, e_param, t_param, count = train(type)
    k = 5 ## top 5 best
    runPart4(type,obs_space, e_param, t_param, count, k)

    # pp.pprint(count)
