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
    for key in pmatrix.keys():
        body = body + 1
        if ptmatrix[key[0], key[1]] == 1:
            supp = supp + 1
    if body == 0:
        SC = 0
    else:
        SC = supp / body
    if head == 0:
        HC = 0
    else:
        HC = supp / head
    return SC, HC


def evaluateAndFilter(pt, p, factdic, minSC, minHC, entitysize):
    # evaluation certain rule
    p1 = p[0]
    p2 = p[1]
    pmatrix = sparse.dok_matrix(np.dot(getmatrix(factdic, p1, entitysize), getmatrix(factdic, p2, entitysize)))
    ptmatrix = getmatrix(factdic, pt, entitysize)
    # calculate the SC and HC
    SC, HC = calSCandHC(pmatrix, ptmatrix)
    if SC > minSC and HC > minHC:
        print("\nThis is " + str(p))
        print("The Standard Confidence of this rule is " + str(SC))
        print("The Head Coverage of this rule is " + str(HC))
        return True
    return False


def learn_weights(fact_dic, candidate, entsize, pt):
    # [[37, 0], [19, 0], [59, 0], [8, 0]]
    rule_Length = 2
    training_Iteration = 10
    learning_Rate = 0.1
    regularization_rate = 0.01

    model = mlw.LearnModel()
    model.__int__(rule_Length, training_Iteration, learning_Rate, regularization_rate, fact_dic, entsize)
    model.load_data(candidate, pt)
    model.train()
    return 0


def save_rules(BENCHMARK, nowPredicate, candidate, model):
    with open("./sampled/" + BENCHMARK + "/relation2id.txt") as f:
        preSize = f.readline()
        pre = [line.strip('\n').split(' ') for line in f.readlines()]
    print("\nThe final rules are:")
    i = 1
    f = open('./rule/' + BENCHMARK + '/rule_After_' + str(model)[15:21] + '.txt', 'a+')
    print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    rule_of_Pt = len(candidate)
    print(str(rule_of_Pt) + "\n")
    f.write("num: " + str(rule_of_Pt) + "\n")
    for rule in candidate:
        print(rule[0])
        print(rule[1])
        line = "Rule " + str(i) + ": " + pre[rule[0]][1] + "  &&  " + pre[rule[1]][1] + "\n"
        print(line)
        f.write(line)
        i = i + 1
    f.write("\n")
    f.close()
    return rule_of_Pt


def searchAndEvaluate(flag, BENCHMARK, nowPredicate, entity, relation, dimension, model):
    relsize = relation.shape[0]
    entsize = entity.shape[0]
    if flag == 0:
        relation = np.reshape(relation, [relsize, dimension, dimension])
    # print(relation.shape)  # (-1, 100, 100) or (-1, 100)
    # print(entity.shape)  # (-1, 100)

    # Score Function
    # calculate the f1
    print("\nBegin to calculate the f1")
    syn = np.zeros(shape=(relsize, relsize))  # normal matrix, because matrix's multiply is not reversible
    # the array's shape is decided by the length of rule, now length = 2
    scorefunction1(flag, syn, nowPredicate[0], relation)
    # calculate the f2
    print("\nBegin to calculate the f2")
    coocc = np.zeros(shape=(relsize, relsize))  # normal matrix
    with open("./sampled/" + BENCHMARK + "/Fact.txt") as f:
        factsSize = f.readline()
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
    # print(facts)
    fact_dic = scorefunction2(coocc, relsize, facts, entity, nowPredicate[0])

    # How to choose this value to get candidate rules?
    candidate = []
    matrics = syn + coocc
    # matrics = coocc  # will be changed!!!!!
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
            minHC = 0.01
            if evaluateAndFilter(nowPredicate[0], max_index, fact_dic, minSC, minHC, entsize):
                candidate.append(max_index)
                constant_flag = False
            else:
                flag = flag + 1
                constant_flag = True
            if flag == 20 and constant_flag is True:
                flag = -1
    print(candidate)
    learn_weights(fact_dic, candidate, entsize, nowPredicate[0])
    rule_of_Pt = save_rules(BENCHMARK, nowPredicate, candidate, model)
    return rule_of_Pt
