import time
import numpy as np
from scipy import sparse
import model_learn_weights as mlw


def sim(para1, para2):  # similarity of vector or matrix
    return np.e ** (-np.linalg.norm(para1 - para2, ord=2))


def scorefunction1(flag, syn, pt, relation):  # synonymy
    for i in range(relation.shape[0]):
        if flag == 0:
            for j in range(relation.shape[0]):
                syn[i][j] = sim(np.dot(relation[i], relation[j]), relation[pt])
        else:
            for j in range(i, relation.shape[0]):
                syn[i][j] = sim(relation[i] + relation[j], relation[pt])
    print("\nf1 matrix: ")
    print(syn)


def scorefunction2(coocc, relsize, facts, entity, pt):  # co-occurrence
    # get the different object and subject for every predicate
    # print(relsize)
    objdic = {}  # key:predicate value: set
    subdic = {}  # key:predicate value: set
    factdic = {}  # key:predicate value: list
    for x in range(facts.shape[0]):
        if facts[x, 2] not in objdic:  # new key
            tempsub = set()
            tempobj = set()
            tempfact = []
        else:
            tempsub = subdic.get(facts[x, 2])
            tempobj = objdic.get(facts[x, 2])
            tempfact = factdic.get(facts[x, 2])
        tempsub.add(facts[x, 0])
        tempobj.add(facts[x, 1])
        tempfact.append(facts[x, :])
        subdic[facts[x, 2]] = tempsub
        objdic[facts[x, 2]] = tempobj
        factdic[facts[x, 2]] = tempfact
    # get the average vector of average predicate which is saved in the dictionary
    average_vector = {}
    for key in subdic:
        # print(key)
        sub = sum([entity[item, :] for item in subdic[key]]) / len(subdic[key])
        obj = sum([entity[item, :] for item in objdic[key]]) / len(objdic[key])
        average_vector[key] = [sub, obj]
    # print("\n the dic's size is equal to the predicates' number! ")
    # print(len(average_vector))
    for i in range(relsize):
        for j in range(relsize):
            coocc[i][j] = sim(average_vector.get(i)[1], average_vector.get(j)[0]) \
                          + sim(average_vector.get(i)[0], average_vector.get(pt)[0]) \
                          + sim(average_vector.get(j)[1], average_vector.get(pt)[1])
    print("\nf2 matrix: ")
    print(coocc)
    return factdic


def getmatrix(factdic, p, entitysize):
    # sparse matrix
    pfacts = factdic.get(p)
    pmatrix = sparse.dok_matrix((entitysize, entitysize), dtype=np.int32)
    for f in pfacts:
        pmatrix[f[0], f[1]] = 1
    return pmatrix


def calSCandHC(pmatrix, ptmatrix):
    # entitysize = pmatrix.shape[0]
    head = len(ptmatrix)
    supp = 0
    body = 0
    # calculate New SC
    supp_score = 0
    body_score = 0
    for key in pmatrix.keys():
        body = body + 1
        body_score = body_score + pmatrix[key[0], key[1]]
        if ptmatrix[key[0], key[1]] == 1:
            supp = supp + 1
            supp_score = supp_score + pmatrix[key[0], key[1]]
    if body == 0:
        SC = 0
    else:
        SC = supp / body
    if head == 0:
        HC = 0
    else:
        HC = supp / head
    if body_score == 0:
        New_SC = 0
    else:
        New_SC = supp_score / body_score
    return New_SC, SC, HC


def evaluateAndFilter(pt, p, factdic, minSC, minHC, entitysize):
    # evaluation certain rule
    p1 = p[0]
    p2 = p[1]
    pmatrix = sparse.dok_matrix(np.dot(getmatrix(factdic, p1, entitysize), getmatrix(factdic, p2, entitysize)))
    ptmatrix = getmatrix(factdic, pt, entitysize)
    # calculate the SC and HC
    NSC, SC, HC = calSCandHC(pmatrix, ptmatrix)
    if SC > minSC and HC > minHC:
        print("\nThis is " + str(p))
        print("The Head Coverage of this rule is " + str(HC))
        print("The Standard Confidence of this rule is " + str(SC))
        print("The NEW Standard Confidence of this rule is " + str(NSC))
        return True
    return False


def learn_weights(fact_dic, candidate, entsize, pt):
    # [[37, 0], [19, 0], [59, 0], [8, 0]]
    rule_Length = 2
    training_Iteration = 50
    learning_Rate = 0.1
    regularization_rate = 0.1

    model = mlw.LearnModel()
    model.__int__(rule_Length, training_Iteration, learning_Rate, regularization_rate, fact_dic, entsize)
    model.load_data(candidate, pt)
    model.train()
    return 0


def save_rules(BENCHMARK, nowPredicate, candidate, model, pre):
    print("\nThe final rules are:")
    i = 1
    f = open('./rule/' + BENCHMARK + '/rule_After_' + str(model)[15:21] + '.txt', 'a+')
    print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    rule_of_Pt = len(candidate)
    f.write("num: " + str(rule_of_Pt) + "\n")
    for rule in candidate:
        line = "Rule " + str(i) + ": " + str(rule[0]) + " " + pre[rule[0]][1] + "  &&  " \
               + str(rule[1]) + " " + pre[rule[1]][1] + "\n"
        print(line)
        f.write(line)
        i = i + 1
    f.write("\n")
    f.close()
    return rule_of_Pt


def get_facts(BENCHMARK, filename):
    with open(filename + BENCHMARK + "/Fact.txt") as f:
        factsSize = f.readline()
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
    return int(factsSize), facts


# Generally, get predicates after sampled.
def get_pre(BENCHMARK):
    with open("./sampled/" + BENCHMARK + "/relation2id.txt") as f:
        preSize = f.readline()
        pre = [line.strip('\n').split(' ') for line in f.readlines()]
    return int(preSize), pre


def get_fact_dic(pre_sample, facts_all):
    fact_dic = {}
    f = len(facts_all)
    p = int(len(pre_sample) / 2)
    for i in range(f):
        for j in range(p):
            if facts_all[i, 2] == int(pre_sample[2*j][2]):
                if int(pre_sample[2*j][0]) in fact_dic.keys():
                    temp_list1 = fact_dic.get(int(pre_sample[2*j][0]))
                    temp_list2 = fact_dic.get(int(pre_sample[2*j+1][0]))
                else:
                    temp_list1 = []
                    temp_list2 = []
                temp_list1.append([facts_all[i, 0], facts_all[i, 1]])
                temp_list2.append([facts_all[i, 1], facts_all[i, 0]])
                fact_dic[int(pre_sample[2*j][0])] = temp_list1
                fact_dic[int(pre_sample[2*j+1][0])] = temp_list2
    # print(fact_dic.keys())
    return fact_dic


def searchAndEvaluate(f, BENCHMARK, nowPredicate, ent_emb, rel_emb, dimension, model, ent_size_all):
    # entsize = ent_emb.shape[0]
    relsize, pre = get_pre(BENCHMARK)
    if f == 0:
        relation = np.reshape(rel_emb, [relsize, dimension, dimension])
    # print(relation.shape)  # (-1, 100, 100) or (-1, 100)
    # print(entity.shape)  # (-1, 100)

    # Score Function
    # calculate the f1
    print("\nBegin to calculate the f1")
    syn = np.zeros(shape=(relsize, relsize))  # normal matrix, because matrix's multiply is not reversible
    # the array's shape is decided by the length of rule, now length = 2
    scorefunction1(f, syn, nowPredicate[0], rel_emb)
    # calculate the f2
    print("\nBegin to calculate the f2")
    coocc = np.zeros(shape=(relsize, relsize))  # normal matrix
    factsSize, facts = get_facts(BENCHMARK, filename="./sampled/")
    # print(facts)
    _fact_dic = scorefunction2(coocc, relsize, facts, ent_emb, nowPredicate[0])

    # get ALL FACTS dictionary!
    fact_size, facts_all = get_facts(BENCHMARK, filename="./benchmarks/")
    t = time.time()
    print("\nGet ALL FACTS dictionary!")
    fact_dic = get_fact_dic(pre, facts_all)
    print("Time: %s \n" % str(time.time()-t))

    # How to choose this value to get candidate rules? Important!
    candidate = []
    print("Begin to get candidate rules.")
    # Method 1: Top ones until it reaches the 100th.
    '''
    # matrics = syn + coocc
    matrics = coocc  # will be changed!!!!!
    flag = 0
    constant_flag = False
    while flag != -1:
        _max_index = np.where(matrics == np.max(matrics))  # maybe return several pairs
        # print(_max_index)
        fir_dim = list(_max_index[0])
        # print(fir_dim)
        sec_dim = list(_max_index[1])
        max_index = []
        for i in range(len(fir_dim)):
            max_index = [fir_dim[i], sec_dim[i]]
            # print(max_index)
            matrics[max_index[0]][max_index[1]] = -1  # set it to the min
            minSC = 0.01
            minHC = 0.001
            if evaluateAndFilter(nowPredicate[0], max_index, fact_dic, minSC, minHC, entsize):
                candidate.append(max_index)
                constant_flag = False
            else:
                flag = flag + 1
                constant_flag = True
            if flag == 100 and constant_flag is True:
                flag = -1
    '''
    # Method 2: Use two matrices to catch rules.
    minSC = 0.01
    minHC = 0.001
    mark_Matrix = np.zeros(shape=(relsize, relsize))
    print(" Begin to use syn.")
    middle_syn = (np.max(syn) - np.min(syn)) * 0.55 + np.min(syn)
    rawrulelist = np.argwhere(syn > middle_syn)
    print(len(rawrulelist))
    # print(rawrulelist)
    for index in rawrulelist:
        if evaluateAndFilter(nowPredicate[0], index, fact_dic, minSC, minHC, ent_size_all):
            candidate.append(index)
            mark_Matrix[index[0], index[1]] = 1
        if evaluateAndFilter(nowPredicate[0], [index[1], index[0]], fact_dic, minSC, minHC, ent_size_all):
            candidate.append(index)
            mark_Matrix[index[0], index[1]] = 1

    print(" Begin to use coocc.")
    middle_coocc = (np.max(coocc) - np.min(syn)) * 0.8 + np.min(syn)
    rawrulelist = np.argwhere(coocc > middle_coocc)
    print(len(rawrulelist))
    # print(rawrulelist)
    for index in rawrulelist:
        if mark_Matrix[index[0], index[1]] == 0:
            if evaluateAndFilter(nowPredicate[0], index, fact_dic, minSC, minHC, ent_size_all):
                candidate.append(index)

    # Evaluation is still a cue method!

    print("\n*^_^* Yeah, there are %d rules. *^_^*\n" % len(candidate))
    # learn_weights(fact_dic, candidate, entsize, nowPredicate[0])  #ent_size_all??? or entsize.
    rule_of_Pt = save_rules(BENCHMARK, nowPredicate, candidate, model, pre)
    return rule_of_Pt
