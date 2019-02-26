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
    fl = np.zeros(factSize, dtype='int32')
    facts = np.c_[facts, fl]
    return facts, entity_size


def first_sample_by_Pt(Pt, facts):
    print("Step 1: First sample by Pt to get E_0:")
    time_start = time.time()
    E_0 = set()
    P_0 = set()
    P_0.add(Pt)
    F_0 = []
    for f in facts:
        if f[2] == Pt and f[3] == 0:
            fact = f.tolist()
            F_0.append(fact)
            f[3] = 1  # Mark it has been included.
            if f[0] not in E_0:
                E_0.add(f[0])
            if f[1] not in E_0:
                E_0.add(f[1])
    print("E_0 size: %d" % len(E_0))
    print("P_0 size: %d" % len(P_0))
    print("F_0 size: %d" % len(F_0))
    time_end = time.time()
    print('Step 1 cost time:', time_end-time_start)
    return E_0, P_0, F_0, facts


def sample_by_i(index, E_i_1_new, facts):
    print("\nStep 2: Sample by %d:" % index)
    time_start = time.time()
    E_i = set()  # Maybe it contains some repeated entities.
    P_i = set()
    F_i_new = []
    flag = 0
    for f in facts:
        if f[0] in E_i_1_new:  # set can not add the same element
            E_i.add(f[1])
            P_i.add(f[2])
            flag = 1
        if f[1] in E_i_1_new:
            E_i.add(f[0])
            P_i.add(f[2])
            flag = 1
        if flag == 1 and f[3] == 0:
            F_i_new.append(f)
            f[3] = 1
        flag = 0
    print("E_%d size: %d (Maybe it contains some repeated entities.)" % (index, len(E_i)))
    print("P_%d size: %d" % (index, len(P_i)))
    print("F_%d_new size: %d" % (index, len(F_i_new)))
    time_end = time.time()
    print('Step 2 cost time:', time_end - time_start)
    return E_i, P_i, F_i_new, facts


def save_and_reindex(length, save_path, E, P, F, Pt, predicate_name, P_list):
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
            f.write(str(2 * i) + "	" + str(name) + "	" + str(pre_sampled_list[i]) + "\n")
            f.write(str(2 * i + 1) + "	" + str(name) + "^-1	" + str(pre_sampled_list[i]) + "\n")
            if pre_sampled_list[i] == curPredicate[0]:
                nowPredicate = [i, name]
        f.close()
    # process the sample predicates' index.
    P_new_index_list = []
    for P_i in P_list:  #P_i is a set.
        P_i_list = []
        for p_old_index in P_i:
            new_index = pre_sampled_list.index(p_old_index)
            P_i_list.append(new_index*2)
            P_i_list.append(new_index*2+1)
        P_new_index_list.append(P_i_list)
        # print(P_i_list)
    # test
    print("after sample, the index:")
    for i in range(len(P_list)):
        print("i = %d, len=%d" % (i, len(P_list[i])))
    print("after sample, the reindex:")
    for i in range(len(P_new_index_list)):
        print("i = %d, len=%d" % (i, len(P_new_index_list[i])))

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
    return nowPredicate, P_new_index_list
