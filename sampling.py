import scipy.io as sio  # related with .mat
import matplotlib.pyplot as plt
import numpy as np
import time


# Because in facts, the index of predicate starts from 1 in DB, but FB15K from 0!
def readData1(BENCHMARK):  # index from 1
    data = sio.loadmat('./benchmarks/' + BENCHMARK + '/tensord.mat')
    print("\nThe variation int the .mat file:")
    print(data.keys())

    # dict_keys(['__header__', '__version__', '__globals__', 'vals', 'rattr', 'eattr', 'subs', 'size'])
    # __header__   'MATLAB 5.0 MAT-file Platform: posix, Created on: Tue Jul 25 00:09:01 2017'
    # __version__  1.0
    vals = data.get('vals')  # all [[1],[1],...]
    rattr = data.get('rattr')  # null
    eattr = data.get('eattr')  # null
    facts = data.get('subs')  # triple?  [[head,tail,predicate],...]
    factSize = len(facts)  # DBpedia: 11024066  一千万条数据
    print("Total facts:" + str(factSize))
    size = data.get('size')  # DBpedia: [head entity size:3102999, tail entity size:3102999, predicate size:650]
    print("Data size:" + str(size))
    return facts, str(size[0]).strip('[').strip(']')


def readData0(BENCHMARK):  # index from 0
    # read the Fact.txt: h t r
    with open('./benchmarks/' + BENCHMARK + '/Fact.txt', 'r') as file:
        factSize = file.readline()
        print("Total facts:" + factSize)
        facts = np.array([line.strip('\n').split(' ') for line in file.readlines()], dtype='int32')
        # print(facts)
    with open('./benchmarks/' + BENCHMARK + '/entity2id.txt', 'r') as entityfile:
        entitysize = entityfile.readline()
        print("Total entities:" + str(entitysize))
    return facts, entitysize


def sample1(BENCHMARK, Pt, predicateName):
    time_start = time.time()
    facts, entitysize = readData1(BENCHMARK)

    # Sampling the Ei and Fi, i<lenOfRules
    lenOfRules = 2
    print("\nStep1: get the E0")
    F0ofPt = []
    E0ofPt = set()
    curPredicate = []
    print(Pt)
    print(predicateName[Pt])
    print("Pt's index:" + str(Pt+1) + "  " + predicateName[Pt])
    curPredicate.append(Pt+1)  # it will change
    curPredicate.append(predicateName[Pt])
    for f in facts:
        fact = f.tolist()
        if f[2] == Pt + 1:  # Because in facts, the index of predicate starts from 1 in DB, but FB15K from 0!
            F0ofPt.append(fact)
            if f[0] not in E0ofPt:
                E0ofPt.add(f[0])
            if f[1] not in E0ofPt:
                E0ofPt.add(f[1])
    factsSizeOfPt = len(F0ofPt)
    entSizeOfPt = len(E0ofPt)
    print("FactsPt 0 size: " + str(factsSizeOfPt))
    print("E0ofPt size: " + str(entSizeOfPt))

    # save in file for every predicate, then it will clear for the next?

    time_1 = time.time()
    print(curPredicate)
    print('Step 1 cost time:', time_1-time_start)

    E1ofPt = set()
    predSampled = set()
    F1ofPt = []
    outFlag = 0
    flag = 0
    print("\nStep2: get the E1")
    for fact in facts:
        if fact[0] in E0ofPt:
            E1ofPt.add(fact[1])  # set can not add the same element
            predSampled.add(fact[2])
            flag = 1
        if fact[1] in E0ofPt:
            E1ofPt.add(fact[0])
            predSampled.add(fact[2])
            flag = 1
        f = fact.tolist()
        if flag == 1 and f not in F0ofPt:
            F1ofPt.append(f)
        flag = 0
        outFlag = outFlag+1
        if outFlag % 100000 == 0:
            print(outFlag)
            print(f)
    print("F1ofPt size: " + str(len(F1ofPt)))
    print("E1ofPt size: " + str(len(E1ofPt)))
    print("Predicates after sampled size:" + str(len(predSampled)))
    time_2 = time.time()
    print('Step 2 cost time:', time_2 - time_1)

    print("\nStep3: get the Union, recode the index and write into file.")
    EofPt = E0ofPt | E1ofPt  # EofPt entity
    F0ofPt.extend(F1ofPt)  # F0ofPt facts
    # predSampled predicates

    # entity
    Entity = np.array(list(EofPt))
    f = open('./sampled/' + BENCHMARK + '/entity2id.txt', 'w')
    entSizeOfPt = len(EofPt)
    f.write(str(entSizeOfPt) + "\n")  # after sampling
    print("EntityOfPt size: " + str(entSizeOfPt))
    for x in range(entSizeOfPt):
        f.write(str(x) + "\n")
    f.close()

    # In this part, after sampling, we should get the reverse predicates and its facts!!!
    # predicate
    f = open('./sampled/' + BENCHMARK + '/relation2id.txt', 'w')
    predSampledli = list(predSampled)
    predSize = len(predSampled)
    nowPredicate = []
    f.write(str(predSize*2) + "\n")  # The number of all relations becomes two times!
    print("Predicated size: " + str(predSize*2))

    #  name ok!
    for i in range(predSize):
        name = predicateName[predSampledli[i]-1]
        f.write(str(2*i) + " " + name + "\n")
        f.write(str(2*i+1) + " " + name + "^-1\n")
        if curPredicate[1] == name:
            nowPredicate = [2*i, name]
    Predicate = np.array(predSampledli)
    # print(predSampled)
    # print(predSampledli)
    # print(Predicate)
    f.close()

    # fact
    Fact = []
    for f in F0ofPt:
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2]))*2])
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2]))*2+1])
    f = open('./sampled/' + BENCHMARK + '/Fact.txt', 'w')
    factsSizeOfPt = len(Fact)
    f.write(str(factsSizeOfPt) + "\n")  # The number of all facts becomes two times!
    print("FactsOfPt size: " + str(factsSizeOfPt))
    for line in Fact:
        f.write(" ".join(str(letter) for letter in line) + "\n")
    f.close()
    # print(Fact)

    time_3 = time.time()
    print('Step 3 cost time:', time_3 - time_2)

    time_end = time.time()
    print('\nTotally cost time:', time_end - time_start)

    print("Pt: ")
    print(nowPredicate)
    return nowPredicate


def sample0(BENCHMARK, Pt, predicateName):
    time_start = time.time()

    facts, entitysize = readData0(BENCHMARK)

    # Sampling the Ei and Fi, i<lenOfRules
    # input: facts, Pt, lenOfRules
    lenOfRules = 2
    print("\nStep1: get the E0")

    F0ofPt = []
    E0ofPt = set()
    curPredicate = []
    print("Pt's index:" + str(Pt) + "  " + str(predicateName[Pt]))
    curPredicate.append(Pt)  # it will change
    curPredicate.append(predicateName[Pt])
    for f in facts:
        fact = f.tolist()
        # return the entities and facts related to certain entity
        # just for the Ei-1 to Ei
        if f[2] == Pt:
            F0ofPt.append(fact)
            if f[0] not in E0ofPt:
                E0ofPt.add(f[0])
            if f[1] not in E0ofPt:
                E0ofPt.add(f[1])
    factsSizeOfPt = len(F0ofPt)
    entSizeOfPt = len(E0ofPt)
    print("FactsPt 0 size: " + str(factsSizeOfPt))  # 35088
    print("E0ofPt size: " + str(entSizeOfPt))  # 45709

    time_1 = time.time()
    print(curPredicate)
    print('Step 1 cost time:', time_1-time_start)

    E1ofPt = set()
    predSampled = set()
    F1ofPt = []
    outFlag = 0
    flag = 0
    print("\nStep2: get the E1")
    for fact in facts:
        if fact[0] in E0ofPt:
            E1ofPt.add(fact[1])  # set can not add the same element
            predSampled.add(fact[2])
            flag = 1
        if fact[1] in E0ofPt:
            E1ofPt.add(fact[0])
            predSampled.add(fact[2])
            flag = 1
        f = fact.tolist()
        if flag == 1 and f not in F0ofPt:
            F1ofPt.append(f)
        flag = 0
        outFlag = outFlag+1
        if outFlag % 100000 == 0:
            print(outFlag)
            print(f)
    print("F1ofPt size: " + str(len(F1ofPt)))
    print("E1ofPt size: " + str(len(E1ofPt)))
    print("Predicates after sampled size:" + str(len(predSampled)))
    time_2 = time.time()
    print('Step 2 cost time:', time_2 - time_1)

    print("\nStep3: get the Union, recode the index and write into file.")
    EofPt = E0ofPt | E1ofPt  # EofPt entity
    F0ofPt.extend(F1ofPt)  # F0ofPt facts
    # predSampled predicates

    # entity
    Entity = np.array(list(EofPt))
    f = open('./sampled/' + BENCHMARK + '/entity2id.txt', 'w')
    entSizeOfPt = len(EofPt)
    f.write(str(entSizeOfPt) + "\n")  # after sampling
    print("EntityOfPt size: " + str(entSizeOfPt))
    for x in range(entSizeOfPt):
        f.write(str(x) + "\n")
    f.close()

    # predicate
    f = open('./sampled/' + BENCHMARK + '/relation2id.txt', 'w')
    predSampledli = list(predSampled)
    predSize = len(predSampled)
    nowPredicate = []
    f.write(str(predSize*2) + "\n")  # after sampling
    print("Predicated size: " + str(predSize*2))

    #  name ok!
    for i in range(predSize):
        name = predicateName[predSampledli[i]]
        f.write(str(2*i) + " " + str(name) + "\n")
        f.write(str(2*i+1) + " " + str(name) + "^-1\n")
        if curPredicate[1] == name:
            nowPredicate = [i, name]
    Predicate = np.array(predSampledli)
    # print(predSampled)
    # print(predSampledli)
    # print(Predicate)
    f.close()

    # fact
    Fact = []
    for f in F0ofPt:
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2]))*2])
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2]))*2+1])
    f = open('./sampled/' + BENCHMARK + '/Fact.txt', 'w')
    factsSizeOfPt = len(Fact)
    f.write(str(factsSizeOfPt) + "\n")
    print("FactsOfPt size: " + str(factsSizeOfPt))
    for line in Fact:
        f.write(" ".join(str(letter) for letter in line) + "\n")
    f.close()

    time_3 = time.time()
    print('Step 3 cost time:', time_3 - time_2)

    time_end = time.time()
    print('\nTotally cost time:', time_end - time_start)
    print("Pt: ")
    print(nowPredicate)
    return nowPredicate
