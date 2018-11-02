import scipy.io as sio  # related with .mat
import time
import numpy as np


# Because in facts, the index of predicate starts from 1 in DB, Yago and Wiki.
def preprocess(BENCHMARK):  # index from 1
    data = sio.loadmat('./benchmarks/' + BENCHMARK + '/tensord.mat')
    print("\nThe variation int the .mat file:")
    print(data.keys())
    facts = data.get('subs')  # triple?  [[head,tail,predicate],...]
    size = data.get('size')  # DBpedia: [head entity size:3102999, tail entity size:3102999, predicate size:650]
    fact_size = str(len(facts))  # DBpedia: 11024066
    entity_size = str(size[0]).strip('[').strip(']')
    relation_size = str(size[2]).strip('[').strip(']')
    print("Fact size:" + fact_size)
    print("Relation size:" + relation_size)
    print("Entity size:" + entity_size)

    # entity
    f = open('./benchmarks/' + BENCHMARK + '/Entity.txt', 'w')
    f.write(entity_size + "\n")
    f.close()

    # relation
    with open('./benchmarks/' + BENCHMARK + '/predindex.txt', 'r') as file:
        predicate_name = [line.strip('\n').strip('[').strip(']').split(',')[1] for line in file.readlines()]
    f = open('./benchmarks/' + BENCHMARK + '/Relation.txt', 'w')
    f.write(relation_size + "\n")
    for line in predicate_name:
        f.write(line + "\n")
    f.close()

    # fact
    f = open('./benchmarks/' + BENCHMARK + '/Fact.txt', 'w')
    f.write(fact_size + "\n")
    for line in facts:
        f.write(str(line[0]-1) + " " + str(line[1]-1) + " " + str(line[2]-1) + "\n")
    f.close()
    print("End pre-processing!")


def read_data(BENCHMARK):  # index from 0
    # read the Fact.txt: h t r
    with open('./benchmarks/' + BENCHMARK + '/Fact.txt', 'r') as file:
        factSize = file.readline()
        print("Total facts:" + factSize)
        facts = np.array([line.strip('\n').split(' ') for line in file.readlines()], dtype='int32')
        # print(facts)
    with open('./benchmarks/' + BENCHMARK + '/Entity.txt', 'r') as entityfile:
        entitysize = entityfile.readline()
        print("Total entities:" + str(entitysize))
    return facts, entitysize


def sample(index_flag, BENCHMARK, Pt, predicateName, ent, rel):
    time_start = time.time()

    facts, entitysize = read_data(BENCHMARK)

    curPredicate = []
    nowPredicate = []
    print("Pt's index:" + str(Pt+index_flag) + " " + predicateName[Pt])  # index
    curPredicate.append(Pt+index_flag)  # index
    curPredicate.append(predicateName[Pt])

    # Sampling the Ei and Fi, i<lenOfRules
    # input: facts, Pt, lenOfRules
    lenOfRules = 2
    print("\nStep1: get the E0")

    F0ofPt = []
    E0ofPt = set()
    for f in facts:
        fact = f.tolist()
        # return the entities and facts related to certain entity
        # just for the Ei-1 to Ei
        if f[2] == Pt+index_flag:  # index
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
    print('Step 1 cost time:', time_1 - time_start)

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
        outFlag = outFlag + 1
        if outFlag % 100000 == 0:
            print(outFlag)
            print(f)
    print("F1ofPt size: " + str(len(F1ofPt)))
    print("E1ofPt size: " + str(len(E1ofPt)))
    print("Predicates after sampled size:" + str(len(predSampled)))
    time_2 = time.time()
    print('Step 2 cost time:', time_2 - time_1)

    print("\nStep3: get the Union, recode the index and get the embedding.")
    EofPt = E0ofPt | E1ofPt  # EofPt entity
    F0ofPt.extend(F1ofPt)  # F0ofPt facts
    # predSampled predicates

    # entity
    entity = list(EofPt)
    entSizeOfPt = len(entity)
    ent_emb = np.zeros(shape=(entSizeOfPt, 100))
    for i in range(entSizeOfPt):
        ent_emb[i, :] = ent[entity[i], :]
    # print(ent_emb)

    # relation
    relation = list(predSampled)
    relSizeOfPt = len(relation)
    rel_emb = np.zeros(shape=(relSizeOfPt, 100))
    for i in range(relSizeOfPt):
        rel_emb[i, :] = rel[relation[i], :]
    # print(rel_emb)

    # save relation with the name to the file
    f = open('./sampled/' + BENCHMARK + '/Relation.txt', 'w')
    f.write(str(relSizeOfPt) + "\n")  # after sampling
    print("Predicated size: " + str(relSizeOfPt))
    for i in range(relSizeOfPt):
        name = predicateName[relation[i]-index_flag]  # index
        f.write(str(i) + " " + name + "\n")
        if curPredicate[1] == name:
            nowPredicate = [i, name]
    Predicate = np.array(relation)
    f.close()

    # fact
    Fact = []
    Entity = np.array(entity)
    for f in F0ofPt:
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2]))])
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

    return ent_emb, rel_emb, nowPredicate






    

