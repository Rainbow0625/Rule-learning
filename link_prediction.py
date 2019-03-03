import numpy as np
import pickle
import time
from scipy import sparse
import gc


def read_data(filename, file_type):
    # read the Fact.txt: h t r
    with open(filename + 'Fact.txt', 'r') as f:
        factSize = int(f.readline())
        facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
        print("Total %s facts:%d" % (file_type, factSize))
    entity_size = 0
    if file_type == "train":
        with open(filename + 'entity2id.txt', 'r') as f:
            entity_size = int(f.readline())
            print("Total %s entities:%d" % (file_type, entity_size))
    return facts, entity_size


def get_pre(benchmark, filename):
    with open(filename + benchmark + "/relation2id.txt") as f:
        preSize = f.readline()
        pre = []
        for line in f.readlines():
            pre.append(line.strip('\n').split("	"))
    return pre


def get_fact_dic(pre, facts_all):
    # Only save once for the reverse pre.e.g. 0, 2, 4....
    # fact_dic: key: P_index_new , value: all_fact_list
    old_index_p = np.array([pre[2] for pre in pre], dtype=np.int32)
    fact_dic = {}
    for f in facts_all:
        if f[2] in set(old_index_p):
            new_index = np.where(old_index_p == f[2])[0][0]  # It must be even.
            if new_index in fact_dic.keys():
                temp_list = fact_dic.get(new_index)
            else:
                temp_list = []
            temp_list.append([f[0], f[1]])
            fact_dic[new_index] = temp_list
    return fact_dic


def get_onehot_matrix(p, fact_dic, ent_size):
    # sparse matrix
    re_flag = False
    if p % 2 == 1:
        p = p - 1
        re_flag = True
    pfacts = fact_dic.get(p)
    pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    if re_flag:
        for f in pfacts:
            pmatrix[f[1], f[0]] = 1
    else:
        for f in pfacts:
            pmatrix[f[0], f[1]] = 1
    return pmatrix


def train_predict(benchmark, pt, pre):
    predict_fact_num = 0
    predict_Qfact_num = 0
    pre_facts_list = []
    with open('./rule/' + benchmark + '/rule_' + str(pt) + '.pk', 'rb') as fp:
        rules = pickle.load(fp)  # [index, flag={1:Rule, 2:Quality Rule}, degree=[SC, HC]]
    # Get fact_dic_all to generate the sparse matrix.
    facts, ent_size = read_data(filename="./benchmarks/" + benchmark + '/' + str(pt) + "/train/", file_type="train")
    # fact_dic: key: P_index_new , value: all_fact_list
    # Note that fact_dic ONLY save once for the reverse pre.e.g. 0, 2, 4....
    fact_dic = get_fact_dic(pre, facts)
    predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_key_by_rule = []
    for rule in rules:
        index = rule[0]
        # degree = rule[2]
        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)
        # Predict num of Qfacts:
        predict_matrix += pmatrix
        predict_key_by_rule.append([list(key) for key in pmatrix.keys()])
        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    for key in predict_matrix.keys():
        fact = list(key)
        mid_SC = 1
        for i_rule in len(predict_key_by_rule):
            if fact in predict_key_by_rule[i_rule]:
                mid_SC *= (1 - rules[i_rule][2][0])
        CD = 1 - mid_SC
        if CD >= 0.9:
            predict_Qfact_num += + 1
            pre_facts_list.append([fact, CD, 1])
        else:
            pre_facts_list.append([fact, CD, 0])
    # Save the fact and CD in file.
    with open('./linkprediction/' + benchmark + '/predict_Pt_' + str(pt) + '.txt', 'w') as f:
        for item in pre_facts_list:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, predict fact num: %d\n" % (pt, predict_fact_num))
        f.write("For Pt: %d, predict Qfact num: %d\n" % (pt, predict_Qfact_num))
    print("For Pt: %d, predict fact num: %d" % (pt, predict_fact_num))
    print("For Pt: %d, predict Qfact num: %d" % (pt, predict_Qfact_num))
    return pre_facts_list


def test(benchmark, pt, pre_facts_list):
    mid_Hits_10 = 0
    mid_MRR = 0
    test_facts, _ = read_data(filename="./benchmarks/" + benchmark + '/' + str(pt) + "/test/", file_type="test")
    # Rank the predicted facts.
    pre_facts_list.sort()
    predicted_facts = [item[0] for item in pre_facts_list]
    test_result = []
    for test_fact in test_facts:
        t = [test_fact[0], test_fact[1]]
        if t in predicted_facts:
            top = predicted_facts.index(t) + 1
            mid_MRR += 1 / top
            Hit = 0
            if top <= 10:
                Hit = 1
                mid_Hits_10 += 1
            test_result.append([t, top, Hit, 1/top])
    Hit_10 = mid_Hits_10/len(test_facts)
    MRR = mid_MRR/len(test_facts)
    # Save the results in file.
    with open('./linkprediction/' + benchmark + '/test_Pt_' + str(pt) + '.txt', 'w') as f:
        for item in test_result:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, Hit_10))
        f.write("For Pt: %d, MRR: %f\n" % (pt, MRR))
    print("For Pt: %d, Hits@10: %f" % (pt, Hit_10))
    print("For Pt: %d, MRR: %f" % (pt, MRR))


if __name__ == '__main__':
    # Run it alone.
    BENCHMARK = "FB15K237"
    begin = time.time()
    pre_sample = get_pre(BENCHMARK, filename="./sampled/")
    Pt_list = [0]  # Randomly select 5.
    for Pt in Pt_list:
        print("Begin to train %d." % Pt)
        pre_facts = train_predict(BENCHMARK, Pt, pre_sample)
        mid = time.time()
        print("Train total time: %f" % mid - begin)
        print('\n')
        print("Begin to test %d." % Pt)
        test(BENCHMARK, Pt, pre_facts)
        mid2 = time.time()
        print("Test total time: %f" % mid2 - mid)
    total_time = time.time() - begin
    print("\nTotal time: %f" % total_time)
    hour = int(total_time / 3600)
    minute = int((total_time - hour * 3600) / 60)
    second = total_time - hour * 3600 - minute * 60
    print(str(hour) + " : " + str(minute) + " : " + str(second))
