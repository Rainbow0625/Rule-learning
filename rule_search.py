import numpy as np
from scipy import sparse
import json


# not in use!
def as_embedding(dct):
    return np.array(dct['ent_embeddings']), np.array(dct['rel_embeddings'])


# not in use!   
def get_embedding(flag, BENCHMARK):
    if flag == 1:
        file = "after"
    elif flag == 0:
        file = "before"
    with open("./embedding/" + file + "/" + BENCHMARK + "/embedding.vec.json") as f:
        return json.loads(f.read(), object_hook=as_embedding)
        # in Python 3.0, the "loads()" function couldn't process the big data!


def sim(para1, para2):  # similarity
    return np.e ** (-np.linalg.norm(para1 - para2))


def scorefunction1(syn, pt, relation):  # synonymy
    for i in range(relation.shape[0]):
        curP = relation[i, :]
        for j in range(i, relation.shape[0]):
            syn[i][j] = sim(curP + relation[j, :], relation[pt, :])
    # print("f1 matrix: ")
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
    # print("f2 matrix: ")
    print(coocc)
    return factdic


def getmatrix(factdic, p, entitysize):
    # sparse matrix
    pfacts = factdic.get(p)
    # print(pfacts)
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


def searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation):
    relsize = relation.shape[0]
    entsize = entity.shape[0]

    # Score Function
    # calculate the f1
    print("\nBegin to calculate the f1")
    syn = np.zeros(shape=(relsize, relsize))  # shang sanjiao matrix
    # the array's shape is decided by the length of rule, now length = 2
    scorefunction1(syn, nowPredicate[0], relation)
    # calculate the f2
    print("\nBegin to calculate the f2")
    coocc = np.zeros(shape=(relsize, relsize))  # normal matrix
    with open("./sampled/" + BENCHMARK + "/Fact.txt") as f:
        factsSize = f.readline()
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
    # print(facts)
    # print(nowPredicate)
    _fact_dic = scorefunction2(coocc, relsize, facts, entity, nowPredicate[0])

    # get ALL FACTS dictionary!
    fact_dic = {}


    # How to choose this value?
    # get candidate rules
    '''
    middle_syn = (np.max(syn) - np.min(syn)) / times_syn + np.min(syn)
    rawrulelist = np.argwhere(syn > middle_syn)
    rulelist = []
    middle_coocc = (np.max(coocc) - np.min(syn)) / times_coocc + np.min(syn)
    for index in rawrulelist:
        if coocc[index[0], index[1]] > middle_coocc:
            rulelist.append(index)
        if coocc[index[1], index[0]] > middle_coocc and index[1] != index[0]:
            rulelist.append([index[1], index[0]])
    print(rulelist)
    '''
    candidate = []
    # matrics = syn + coocc
    matrics = coocc
    flag = 0
    constant_flag = False
    while flag != -1:
        _max_index = np.where(matrics == np.max(matrics))  # maybe return several pairs
        print(_max_index)
        fir_dim = list(_max_index[0])
        print(fir_dim)
        sec_dim = list(_max_index[1])
        max_index = []
        for i in range(len(fir_dim)):
            max_index = [fir_dim[i], sec_dim[i]]
            print(max_index)
            matrics[max_index[0]][max_index[1]] = -1  # set it to the min
            if evaluateAndFilter(nowPredicate[0], max_index, fact_dic, minSC, minHC, entsize):
                candidate.append(max_index)
                constant_flag = False
            else:
                flag = flag + 1
                constant_flag = True
            if flag == 10 and constant_flag == True:
                flag = -1
    print(candidate)

    return candidate
