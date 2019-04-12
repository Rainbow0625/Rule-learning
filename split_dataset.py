import shutil
import numpy as np
import random
import os
# split the Pt's facts to 30%(up to 5K facts) and 70%.
# It is necessary to ensure that each training set contains all the entities, otherwise it is not easy to calculate.


def read_data(filename):
    # read the Fact.txt: h t r
    with open(filename + 'Fact.txt', 'r') as fp:
        factSize = int(fp.readline())
        facts = np.array([line.strip('\n').split(' ') for line in fp.readlines()], dtype='int32')
        print("Total facts:%d" % factSize)
    with open(filename + 'entity2id.txt', 'r') as fp2:
        entity_size = int(fp2.readline())
        print("Total entities:%d" % entity_size)
    with open(filename + "relation2id.txt") as fp:
        preSize = fp.readline()
    return facts, entity_size, preSize


def get_fact_dic(facts_all, Pts):
    # Save for the predicates in Pt_list.
    # fact_dic: key: P_index, value: [all_fact_list].
    facts_dic = {}
    for ff in facts_all:
        if ff[2] in Pts:
            if ff[2] in facts_dic.keys():
                temp_list = facts_dic.get(ff[2])
            else:
                temp_list = []
            temp_list.append([ff[0], ff[1]])
            facts_dic[ff[2]] = temp_list
    return facts_dic


if __name__ == '__main__':
    BENCHMARK = "FB75K"
    FILENAME = "./benchmarks/" + BENCHMARK + '/'
    # Pt_list = [i for i in range(13)]  # Need to create the file to save samples.
    Pt_list = [1]
    print("Split index of predicate:")
    print(Pt_list)
    facts, ent_size, pre_size = read_data(FILENAME)
    fact_dic = get_fact_dic(facts, Pt_list)
    for pt in fact_dic.keys():
        Pt_facts = fact_dic.get(pt)
        Pt_facts_size = len(Pt_facts)
        # split the Pt'Pt_factss facts to 30%(up to 5K facts) and 70%.
        test_num = int(0.3 * Pt_facts_size)
        if test_num > 5000:
            test_num = 5000
        train_num = Pt_facts_size - test_num
        test_facts_index = random.sample([i for i in range(Pt_facts_size)], test_num)
        test_facts = []
        train_facts = []
        for i in range(Pt_facts_size):
            if i in test_facts_index:
                # test_facts: [ent_1, ent_2]  (no predicate)
                test_facts.append(Pt_facts[i])
            else:
                # train_facts: [ent_1, ent_2, pt]  (has predicate)
                train_facts.append([Pt_facts[i][0], Pt_facts[i][1], pt])
        # Save tests in file.
        with open("./benchmarks/" + BENCHMARK + '/' + str(pt) + "/test/Fact.txt", 'w') as f:
            f.write(str(test_num)+'\n')
            for fact in test_facts:
                f.write("%d %d\n" % (fact[0], fact[1]))
        # Save train facts in file. (It still need to save facts of other Pts.)
        rest_ent = set()
        with open("./benchmarks/" + BENCHMARK + '/' + str(pt) + "/train/Fact.txt", 'w') as f:
            for fact in train_facts:
                f.write("%d %d %d\n" % (fact[0], fact[1], fact[2]))
                rest_ent.add(fact[0])
                rest_ent.add(fact[1])
            for fact in facts:  # facts: array
                if fact[2] != pt:
                    f.write("%d %d %d\n" % (fact[0], fact[1], fact[2]))
                    train_num += 1
                    rest_ent.add(fact[0])
                    rest_ent.add(fact[1])
        with open("./benchmarks/" + BENCHMARK + '/' + str(pt) + "/train/Fact.txt", 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(train_num) + '\n' + content)
        # with open("./benchmarks/" + BENCHMARK + '/' + str(pt) + "/train/entity2id.txt", 'w') as f:
            # f.write(str(len(rest_ent))+'\n')
        if test_num > 100:
            print("Split data for %d, test_num: %d" % (pt, test_num))
            Pt_list.append(pt)
        else:
            print("Split data for %d, test_num: %d, too small!" % (pt, test_num))
            shutil.rmtree("./benchmarks/" + BENCHMARK + '/' + str(pt))  # Delete all the file tree.
