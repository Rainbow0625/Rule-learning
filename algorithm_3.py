# -*- coding: utf-8 -*
import time
import sampling_3 as s
from models import TransE, TransD, TransH, TransR, RESCAL
import train_embedding as te
import rule_search_and_learn_weights_2 as r

'''
import sys
sys.stdout.write('\r'+str())
sys.stdout.flush()
'''

BENCHMARK = "FB15K237"
IsUncertain = False
R_minSC = 0.01
R_minHC = 0.001
QR_minSC = 0.5
QR_minHC = 0.001
DEGREE = [R_minSC, R_minHC, QR_minSC, QR_minHC]
Max_rule_length = 3  # not include head atom

# embedding model parameters
work_threads = 5
nbatches = 150
margin = 1  # the margin for the loss function
train_times = 5  # 1000
dimension = 50  # 50
alpha = 0.01  # learning rate
lmbda = 0.01  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
model = TransE.TransE
# Vetor:DistMult.DistMult HolE.HolE
#       TransE.TransE
# Unknown: TransD.TransD TransH.TransH TransR.TransR
# Matrix: RESCAL.RESCAL


def save_rules(rule_length, nowPredicate, candidate, pre):
    print("\nThe final rules are:")
    i = 1
    f = open('./rule/' + BENCHMARK + '/rule_' + str(model)[15:21] + '.txt', 'a+')
    print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    f.write("length: %d, num: %d\n" % (rule_length, len(candidate)))
    R_num = 0
    QR_num = 0
    for rule in candidate:
        index = rule[0]
        flag = rule[1]
        line = ""
        if flag == 1:
            R_num = R_num + 1
            line = "Rule " + str(i) + ": "
        elif flag == 2:
            QR_num = QR_num + 1
            line = "Qualify Rule " + str(i) + ": "
        # output should be modified!
        for j in range(rule_length):
            line = line + str(index[j]) + " " + pre[index[j]][1] + "; "
        line = line + "\n"
        print(line)
        f.write(line)
        i = i + 1
    f.write("\nRules number: %d\n" % R_num)
    f.write("Qualify Rules number: %d\n" % QR_num)
    f.close()


if __name__ == '__main__':
    begin = time.time()
    print("\nThe benchmark is " + BENCHMARK + ".")
    with open('./benchmarks/' + BENCHMARK + '/relation2id.txt', 'r') as f:
        predicateSize = int(f.readline())
        predicateName = [relation.split("	")[0] for relation in f.readlines()]
        print("Total predicates:%d" % predicateSize)

    # get ALL FACTS dictionary!
    rsalw = r.RSALW()
    fact_size, facts_all = rsalw.get_facts(BENCHMARK, filename="./benchmarks/")
    t = time.time()
    print("\nGet ALL FACTS dictionary!")
    _p, pre = rsalw.get_pre(BENCHMARK)
    fact_dic = rsalw.get_fact_dic(pre, facts_all, IsUncertain)
    print("Time: %s \n" % str(time.time() - t))

    # 0:matrix 1:vector
    total_num_rule = 0
    total_time = 0
    for Pt in range(predicateSize):
        # 对于每个规则长度进行循环!
        Pt_start = time.time()
        Pt_i_1 = Pt_start
        print("\n##Begin to sample##\n")
        # after sample by Pt, return the E_i-1 and F_i-1.
        E_i_1, P_i_1, F_i_1, F_rest, ent_size_all = s.first_sample_by_Pt(BENCHMARK, Pt)
        # initialization variable.
        E = E_i_1
        P = P_i_1
        F = F_i_1
        E_i_1_new = E_i_1
        save_path = './sampled/' + BENCHMARK
        num_rule = 0
        for length in range(1, Max_rule_length):
            for i in range(length-1, length):  # only run once.
                # Sample stage.
                E_i, P_i, F_i_new, F_rest_new = s.sample_by_length(i+1, E_i_1_new, F_rest)
                # get the next cycle's variable.
                E_i_1_new = E_i - E_i_1
                F_rest = F_rest_new
                # merge the result
                E = E | E_i
                P = P | P_i
                F.extend(F_i_new)
            nowPredicate = s.save_and_reindex(length+1, save_path, E, P, F, Pt, predicateName)
            print("\n##End to sample##\n")

            print("\n##Begin to train embedding##\n")
            # The parameter of model should be adjust to the best parameters!!!!!!
            ent_emb, rel_emb = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda,
                                             bern, margin, model)
            print("\n##End to train embedding##\n")

            print("\n##Begin to search and evaluate##\n")
            # this part should be modified!!!!!

            candidate = rsalw.search_and_evaluate(1, length+1, BENCHMARK, nowPredicate, ent_emb, rel_emb,
                                                  dimension, ent_size_all, fact_dic, DEGREE, IsUncertain)
            print("\n##End to search and evaluate##\n")

            save_rules(length+1, nowPredicate, candidate, pre)  # i+1:rule length.
            num_rule = num_rule + len(candidate)
            Pt_i = time.time()
            print("Length = %d, Time = %f" % (length+1, (Pt_i-Pt_i_1)))
            Pt_i_1 = Pt_i
        total_num_rule = total_num_rule + num_rule
        Pt_end = time.time()
        Pt_time = Pt_end - Pt_start
        total_time = total_time + Pt_time
        print("This %d th predicate's total time: %f\n" % (Pt, Pt_time))
        print("Until now, all %d predicates' average time: %f\n" % (Pt, total_time/(Pt+1)))
        break
    with open('./rule/'+BENCHMARK+'/rule_' + str(model)[15:21]+'.txt', 'a+') as f:
        f.write("Embedding parameter:\n")
        f.write("model: %s\n" + str(model))
        f.write("train_times: %d\n" % train_times)
        f.write("dimension: %d\n" % dimension)
        f.write("alpha: %f\n" % alpha)
        f.write("lmbda: %f\n" % lmbda)
        f.write("bern: %f\n" % bern)
        f.write("\nR_minSC:%f, R_minHC:%f\n" % (R_minSC, R_minHC))
        f.write("QR_minSC:%f, QR_minHC:%f\n" % (QR_minSC, QR_minHC))
        f.write("Total number of rules: %d\n" % total_num_rule)
        f.write("Average time:%s\n" % str(total_time/int(predicateSize)))  # unless it runs to end.
        f.close()
    end = time.time()-begin
    hour = int(end/3600)
    minute = int((end-hour*3600)/60)
    second = end-hour*3600-minute*60
    print("\nAlgorithm total time: %f" % end)
    print(str(hour)+" : "+str(minute)+" : "+str(second))
