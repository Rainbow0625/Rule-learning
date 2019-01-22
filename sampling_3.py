import numpy as np
import time


def read_data(BENCHMARK):  # index from 0
    # read the Fact.txt: h t r
    with open('./benchmarks/' + BENCHMARK + '/Fact.txt', 'r') as f:
        factSize = int(f.readline())
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
        f.close()
        print("(Before sample, total facts:%d)" % factSize)
    with open('./benchmarks/' + BENCHMARK + '/entity2id.txt', 'r') as f:
        entity_size = int(f.readline())
        f.close()
        print("(Before sample, total entities:%d)" % entity_size)
    return facts, entity_size


def first_sample_by_Pt(BENCHMARK, Pt):
    print("Step 1: First sample by Pt to get E_0:")
    time_start = time.time()
    facts, entitysize = read_data(BENCHMARK)
    E_0 = set()
    P_0 = {Pt}
    F_0 = []
    F_rest = []
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
    print("E_0 size: %d" % len(E_0))
    print("P_0 size: %d" % len(P_0))
    print("F_0 size: %d" % len(F_0))
    print("F_rest size: %d" % len(F_rest))
    time_end = time.time()
    print('Step 1 cost time:', time_end-time_start)
    return E_0, P_0, F_0, F_rest, entitysize


def sample_by_length(index, E_i_1_new, F_rest):
    print("\nStep 2: Sample by length get E_%d:" % index)
    time_start = time.time()
    E_i = set()  # Maybe it contains some repeated entities.
    P_i = set()
    F_i_new = []
    F_rest_new = []
    flag = 0
    for f in F_rest:
        if f[0] in E_i_1_new:  # set can not add the same element
            E_i.add(f[1])
            P_i.add(f[2])
            flag = 1
        if f[1] in E_i_1_new:
            E_i.add(f[0])
            P_i.add(f[2])
            flag = 1
        if flag == 1:
            F_i_new.append(f)
        else:
            F_rest_new.append(f)
        flag = 0
    print("E_%d size: %d (Maybe it contains some repeated entities.)" % (index, len(E_i)))
    print("P_%d size: %d" % (index, len(P_i)))
    print("F_%d_new size: %d" % (index, len(F_i_new)))
    print("F_rest size: %d" % len(F_rest_new))
    time_end = time.time()
    print('Step 2 cost time:', time_end - time_start)
    return E_i, P_i, F_i_new, F_rest_new


def save_and_reindex(length, save_path, E, P, F, Pt, predicate_name):
    print("\nFinal Step:save and reindex, length = %d:" % length)
    curPredicate = [Pt, predicate_name[Pt]]

    # Entity
    with open(save_path + '/entity2id.txt', 'w') as f:
        ent_size = len(E)
        f.write(str(ent_size) + "\n")
        print("  entity size: %d" % ent_size)
        for x in range(ent_size):
            f.write(str(x) + "\n")
        f.close()

    # Predicate: need to add R^-1.
    nowPredicate = []
    with open(save_path + '/relation2id.txt', 'w') as f:
        pre_size = len(P)
        f.write(str(pre_size * 2) + "\n")  # after sampling
        print("  predicate size: %d" % (pre_size * 2))
        pre_sampled_list = list(P)
        for i in range(pre_size):
            name = predicate_name[pre_sampled_list[i]]
            # Note that pre_sampled_list[i] is the old index!
            f.write(str(2 * i) + " " + str(name) + " " + str(pre_sampled_list[i]) + "\n")
            f.write(str(2 * i + 1) + " " + str(name) + "^-1 " + str(pre_sampled_list[i]) + "\n")
            if pre_sampled_list[i] == curPredicate[0]:
                nowPredicate = [i, name]

        f.close()

    # Fact: need to double.
    Fact = []
    Entity = np.array(list(E))
    Predicate = np.array(pre_sampled_list)
    for f in F:
        Fact.append([int(np.argwhere(Entity == f[0])), int(np.argwhere(Entity == f[1])),
                     int(np.argwhere(Predicate == f[2])) * 2])
        Fact.append([int(np.argwhere(Entity == f[1])), int(np.argwhere(Entity == f[0])),
                     int(np.argwhere(Predicate == f[2])) * 2 + 1])
    with open(save_path + '/Fact.txt', 'w') as f:
        factsSizeOfPt = len(Fact)
        f.write(str(factsSizeOfPt) + "\n")
        print("  fact size: " + str(factsSizeOfPt))
        for line in Fact:
            f.write(" ".join(str(letter) for letter in line) + "\n")
        f.close()

    # reindex step:
    print('old:%d new:%d' % (curPredicate[0], nowPredicate[0]))
    print("Pt's original index -- %d : %s" % (curPredicate[0], curPredicate[1]))
    print("Pt's new index -- %d : %s" % (nowPredicate[0], nowPredicate[1]))
    return nowPredicate
