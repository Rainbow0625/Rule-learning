import numpy as np
import time
'''
For the sampling process, RLvLR picked at most 50 neighbours of an entity
 and set the maximum size of each sample to 800 entities. 
'''


def read_data(BENCHMARK, filename):  # index from 0
    # read the Fact.txt: h t r
    with open(filename + BENCHMARK + '/Fact.txt', 'r') as f:
        factSize = int(f.readline())
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
        f.close()
        print("(Before sample, total facts:%d)" % factSize)
    with open(filename + BENCHMARK + '/entity2id.txt', 'r') as f:
        entity_size = int(f.readline())
        f.close()
        print("(Before sample, total entities:%d)" % entity_size)
    fl = np.zeros(factSize, dtype='int32')
    facts = np.c_[facts, fl]
    return facts, entity_size


def get_pre(BENCHMARK, filename):
    with open(filename + BENCHMARK + "/relation2id.txt") as f:
        preSize = int(f.readline())
        pre = []
        for line in f.readlines():
            pre.append(line.strip('\n').split("	"))
    return pre, preSize


def first_sample_by_Pt(Pt, facts):
    print("Step 1: First sample by Pt to get E_0:")
    time_start = time.time()
    E_0_all = set()
    E_0 = set()
    P_0 = set()
    P_0.add(Pt)
    F_0 = []
    for f in facts:
        # Sample.
        if f[2] == Pt and f[3] == 0:
            E_0_all.add(f[0])
            E_0_all.add(f[1])
            if len(E_0) < 5000:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                fact = f.tolist()
                F_0.append(fact)
                f[3] = 1  # Mark it has been included.
                E_0.add(f[0])
                E_0.add(f[1])
            # else:
            #     break
    print("E_0_all size: %d" % len(E_0_all))
    print("E_0 size: %d" % len(E_0))
    print("P_0 size: %d" % len(P_0))
    print("F_0 size: %d" % len(F_0))
    time_end = time.time()
    print('Step 1 cost time:', time_end-time_start)
    return E_0, P_0, F_0, facts, E_0_all


def sample_by_i(index, E_i_1_new, facts):
    print("\nStep 2: Sample by %d:" % index)
    time_start = time.time()
    del_flag = 0
    E_i = set()  # Maybe it contains some repeated entities.
    F_i_new = []
    P_count = {}  # After filtering:   Key: p's index; Value: [count]
    # Statistical occurrences.
    P_dic = {}  # Key: p's index; Value: [count, [fact's index list]]
    for i in range(len(facts)):
        f = list(facts[i])
        if f[0] in E_i_1_new or f[1] in E_i_1_new:
            if f[2] in P_dic.keys():
                value = P_dic.get(f[2])
                # Restrict the number of entity!
                #if len(value[1]) <= 50:   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #    value[1].append(i)  # else, only count the freq.
                value[1].append(i)
                P_dic[f[2]] = [value[0]+1, value[1]]
            else:
                P_dic[f[2]] = [1, [i]]

    keys = list(P_dic.keys())
    for key in keys:
        value = P_dic[key]
        # Pick out predicates with high frequency of occurrence.
        if value[0] > 5000:   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            del_flag = del_flag + 1
        else:
            P_count[key] = value[0]
            # Get E_i and P_count.
            for j in value[1]:
                E_i.add(facts[j][0])
                E_i.add(facts[j][1])
                if facts[j][3] == 0:
                    F_i_new.append(list(facts[j]))
                    facts[j][3] = 1
    P_i = set(P_count.keys())
    print("Count list:")
    print(list(P_count.values()))
    count_mean = int(np.mean(np.array(list(P_count.values()), dtype=np.int32)))
    print("count_mean: %d" % count_mean)
    print("Leave pre num:%d" % len(P_i))
    print("Delete pre num:%d \n" % del_flag)
    print("E_%d size: %d (Maybe it contains some repeated entities.)" % (index, len(E_i)))
    print("P_%d size: %d" % (index, len(P_i)))
    print("F_%d_new size: %d" % (index, len(F_i_new)))
    time_end = time.time()
    print('Step 2 cost time:', time_end - time_start)
    return E_i, P_i, F_i_new, facts, P_count


def filter_predicates_by_count(Pt, P_count_dic, P_new_index_list, fact_dic_sample, fact_dic_all):
    del_flag = 0
    keys = list(P_count_dic.keys())
    for key in keys:
        if P_count_dic.get(key) > 300 or P_count_dic.get(key) < 100:
        if key == Pt or key-1 == Pt:
            continue
        if P_count_dic.get(key) > 250 or P_count_dic.get(key) < 150:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Remove the elements filtered.
            P_new_index_list[-1].remove(key)
            if key in P_new_index_list[-2]:
                P_new_index_list[-2].remove(key)
            if key % 2 == 0:
                if key in fact_dic_all.keys():
                    del fact_dic_all[key]
                    del fact_dic_sample[key]
            else:
                if key-1 in fact_dic_all.keys():
                    del fact_dic_all[key-1]
                    del fact_dic_sample[key-1]
            del_flag = del_flag + 1
    print("Remove num: %d" % del_flag)
    return P_new_index_list, fact_dic_sample, fact_dic_all


def save_and_reindex(length, save_path, E, P, F, Pt, predicate_name, P_i_list, P_count_old):
    print("\nFinal Step:save and reindex, length = %d:" % length)
    old_index_Pt = [Pt, predicate_name[Pt]]

    # Entity
    with open(save_path + '/entity2id.txt', 'w') as f:
        ent_size = len(E)
        f.write(str(ent_size) + "\n")
        print("  Entity size: %d" % ent_size)
        for x in range(ent_size):
            f.write(str(x) + "\n")

    # Predicate: need to add R^-1.
    new_index_Pt = []
    pre_sampled_list = list(P)
    with open(save_path + '/relation2id.txt', 'w') as f:
        pre_size = len(P)
        f.write(str(pre_size * 2) + "\n")  # after sampling
        print("  Predicate size: %d" % (pre_size * 2))
        for i in range(pre_size):
            name = predicate_name[pre_sampled_list[i]]
            # Note that pre_sampled_list[i] is the old index!
            f.write(str(2 * i) + "	" + str(name) + "	" + str(pre_sampled_list[i]) + "\n")
            f.write(str(2 * i + 1) + "	" + str(name) + "^-1	" + str(pre_sampled_list[i]) + "\n")
            if pre_sampled_list[i] == old_index_Pt[0]:
                new_index_Pt = [2 * i, name]  # OMG.
    # Process the sample predicates' index.
    P_i_list_new = []
    for P_i in P_i_list:  # P_i is a set.
        P_i_new = []
        for p_old_index in P_i:
            new_index = pre_sampled_list.index(p_old_index)
            P_i_new.append(new_index*2)
            P_i_new.append(new_index*2+1)
        P_i_list_new.append(P_i_new)
    # test
    print("after sample, the index:")
    for i in range(len(P_i_list)):
        print("i = %d, len=%d" % (i, len(P_i_list[i])))
    print("after sample, the reindex:")
    for i in range(len(P_i_list_new)):
        print("i = %d, len=%d" % (i, len(P_i_list_new[i])))
    # Update the P_count_dic's index to new.
    P_count_new = {}
    for old_index in P_count_old.keys():
        new_index = pre_sampled_list.index(old_index)
        P_count_new[new_index * 2] = P_count_old.get(old_index)
        P_count_new[new_index * 2 + 1] = P_count_old.get(old_index)

    # Fact: need to double.
    facts_sample = np.zeros(shape=(1, 3), dtype=np.int32)
    Entity = np.array(list(E))
    Predicate = np.array(pre_sampled_list)
    for f in F:
        facts_sample = np.r_[facts_sample, np.array([[int(np.argwhere(Entity == f[0])),
                                                     int(np.argwhere(Entity == f[1])),
                                                     int(np.argwhere(Predicate == f[2])) * 2],
                                                     [int(np.argwhere(Entity == f[1])),
                                                      int(np.argwhere(Entity == f[0])),
                                                      int(np.argwhere(Predicate == f[2])) * 2 + 1]],
                                                    dtype=np.int32)]
    facts_sample = np.delete(facts_sample, 0, axis=0)
    with open(save_path + '/Fact.txt', 'w') as f:
        factsSizeOfPt = len(facts_sample)
        f.write(str(factsSizeOfPt) + "\n")
        print("  Fact size: " + str(factsSizeOfPt))
        for line in facts_sample:
            f.write(" ".join(str(letter) for letter in line) + "\n")

    # reindex step:
    print('old:%d new:%d' % (old_index_Pt[0], new_index_Pt[0]))
    print("Pt's old index -- %d : %s" % (old_index_Pt[0], old_index_Pt[1]))
    print("Pt's new index -- %d : %s" % (new_index_Pt[0], new_index_Pt[1]))
    return new_index_Pt, P_i_list_new, P_count_new, facts_sample
