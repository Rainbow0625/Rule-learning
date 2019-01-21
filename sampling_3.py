import scipy.io as sio  # related with .mat
import numpy as np
import time


def readData(BENCHMARK):  # index from 0
    # read the Fact.txt: h t r
    with open('./benchmarks/' + BENCHMARK + '/Fact.txt', 'r') as file:
        factSize = file.readline()
        print("Total facts:" + factSize)
        facts = np.array([line.strip('\n').split(' ') for line in file.readlines()], dtype='int32')
        # print(facts)
    with open('./benchmarks/' + BENCHMARK + '/entity2id.txt', 'r') as entityfile:
        entitysize = entityfile.readline()
        print("Total entities:" + str(entitysize))
    return facts, int(entitysize)


def first_sample_by_Pt(BENCHMARK, Pt, predicateName):
    print("\nFirst sample by Pt to get E_0:")
    time_start = time.time()
    facts, entitysize = readData(BENCHMARK)
    E_0 = set()
    P_0 = [Pt]
    F_0 = []
    F_rest = []
    curPredicate = []
    curPredicate.append(Pt)  # Note that its index will change after sample.
    curPredicate.append(predicateName[Pt])
    print("Pt's original index: %d : %s" % (curPredicate[0], curPredicate[1]))
    for f in facts:
        fact = f.tolist()
        if f[2] == Pt:
            F_0.append(fact)
            if f[0] not in E_0:
                E_0.add(f[0])
            if f[1] not in E_0:
                E_0.add(f[1])
        else:
            F_rest.append(fact)
    print("F_0 size: %d" % len(F_0))  
    print("E_0 size: " % len(E_0)) 
    time_end = time.time()
    print('Step 1 cost time:', time_end-time_start)
    return E_0, P_0, F_0, F_rest


def sample(BENCHMARK, Pt, predicateName):
    time_start = time.time()

    facts, entitysize = readData(BENCHMARK)

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
        # predSampledli[i] is the old index!
        f.write(str(2*i) + " " + str(name) + " " + str(predSampledli[i]) + "\n")
        f.write(str(2*i+1) + " " + str(name) + "^-1 " + str(predSampledli[i]) + "\n")
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
        Fact.append([int(np.argwhere(Entity == f[1])), int(np.argwhere(Entity == f[0])),
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
    return entitysize, nowPredicate
