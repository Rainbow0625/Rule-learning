import numpy as np
import sys
import time
from scipy import sparse
import sampling_3 as s
import gc


def get_fact_dic(pre_sample_of_Pt, facts_all):
    old_pre = set()
    for i in range(len(pre_sample_of_Pt)):
        new_pre_sample = pre_sample_of_Pt[i]
        for p in new_pre_sample:
            if p[2] not in old_pre:
                old_pre.add(p[2])
    # Only save once for the pre.
    # fact_dic: key: P_index_old , value: all_fact_list
    fact_dic = {}
    for f in facts_all:
        if f[2] in old_pre:
            if f[2] in fact_dic.keys():
                temp_list = fact_dic.get(f[2])
            else:
                temp_list = []
            temp_list.append([f[0], f[1]])
            fact_dic[f[2]] = temp_list
    return fact_dic


def get_onehot_matrix(p, fact_dic, ent_size, pre_sample):
    # sparse matrix
    re_flag = False if p % 2 == 0 else True
    new_p = np.array([pre[0] for pre in pre_sample], dtype=np.int32)
    pfacts = fact_dic.get(pre_sample[np.where(new_p == p)[0][0]][2])
    pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    if re_flag:
        for f in pfacts:
            pmatrix[f[1], f[0]] = 1
    else:
        for f in pfacts:
            pmatrix[f[0], f[1]] = 1
    return pmatrix


def predict(lp_save_path, pt, pre_sample_of_Pt, rules, facts, ent_size):
    # rules: [index, flag={1:Rule, 2:Quality Rule}, degree=[SC, HC]]
    predict_fact_num = 0
    predict_Qfact_num = 0
    pre_facts_list = []

    # Get fact_dic_all to generate the sparse matrix.
    # fact_dic: key: P_index_old , value: all_fact_list
    # So NOTE that fact_dic ONLY save once for the pre.
    fact_dic = get_fact_dic(pre_sample_of_Pt, facts)
    predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)

    predict_key_by_rule = []
    i = 0
    print("Predict by rule.")
    for rule in rules:
        sys.stdout.write('\rProgress: %d - %d ' % (i, len(rules)))
        sys.stdout.flush()
        i +=1
        index = rule[0]
        # degree = rule[2]  # why not use degree?
        length = len(index)
        pre_sample = pre_sample_of_Pt[0] if length == 2 else pre_sample_of_Pt[1]
        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size, pre_sample)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, pre_sample))
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
    if len(rules) == len(predict_key_by_rule):
        print("Ok, Next step！")
    i = 0
    print("Calculate the CD.")
    for key in predict_matrix.keys():
        sys.stdout.write('\rProgress: %d - %d ' % (i, len(predict_matrix.keys())))
        sys.stdout.flush()
        i += 1
        fact = list(key)
        mid_SC = 1
        for i_rule in len(predict_key_by_rule):
            if fact in predict_key_by_rule[i_rule]:
                mid_SC *= (1 - rules[i_rule][2][0])
        CD = 1 - mid_SC
        if CD >= 0.9:
            predict_Qfact_num += 1
            pre_facts_list.append([fact, CD, 1])
        else:
            pre_facts_list.append([fact, CD, 0])
    # Save the fact and CD in file.
    with open(lp_save_path + 'predict_Pt_' + str(pt) + '.txt', 'w') as f:
        for item in pre_facts_list:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, predict fact num: %d\n" % (pt, predict_fact_num))
        f.write("For Pt: %d, predict Qfact num: %d\n\n" % (pt, predict_Qfact_num))
    print("For Pt: %d, predict fact num: %d" % (pt, predict_fact_num))
    print("For Pt: %d, predict Qfact num: %d" % (pt, predict_Qfact_num))
    return pre_facts_list, predict_fact_num, predict_Qfact_num


def test(lp_save_path, pt, pre_facts_list):
    mid_Hits_10 = 0
    mid_MRR = 0
    test_facts, _ = s.read_data(filename=lp_save_path + str(pt) + "/test/", file_type="test")
    # Rank the predicted facts by CD.
    pre_facts_list.sort(key=lambda x: x[1], reverse=True)
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
    with open(lp_save_path + 'test_Pt_' + str(pt) + '.txt', 'w') as f:
        for item in test_result:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, Hit_10))
        f.write("For Pt: %d, MRR: %f\n" % (pt, MRR))
    print("For Pt: %d, Hits@10: %f" % (pt, Hit_10))
    print("For Pt: %d, MRR: %f" % (pt, MRR))
    return MRR, Hit_10
