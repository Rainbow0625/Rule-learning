# -*- coding: utf-8 -*
import time
import sampling_3 as s
from models import TransE, TransD, TransH, TransR, RESCAL
import train_embedding as te
import rule_search_and_learn_weights_2 as r
import gc
import send_process_report_email
import pickle
import numpy as np
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
_syn = 800
_coocc = 800

# embedding model parameters
work_threads = 5
nbatches = 150
margin = 1  # the margin for the loss function
train_times = 100  # 1000
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
    print("The final rules :")
    f = open('./rule/' + BENCHMARK + '/rule_' + str(model)[15:21] + '.txt', 'a+')
    # print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    f.write("length: %d, num: %d\n" % (rule_length, len(candidate)))
    R_num = 0
    QR_num = 0
    i = 1
    for rule in candidate:
        index = rule[0]
        flag = rule[1]
        degree = str(rule[2])
        if flag == 1:
            R_num = R_num + 1
            title = "Rule " + str(i) + ": "
        else:
            QR_num = QR_num + 1
            title = "Qualify Rule " + str(i) + ": "
        line = title + " " + str(index) + " :[NSC, SC, HC] " + degree + " "
        for j in range(rule_length):
            line = line + str(index[j]) + " " + pre[index[j]][1] + "; "
        line = line + "\n"
        # print(line)
        f.write(line)
        i = i + 1
    print("\nRules number: %d\n" % R_num)
    print("Qualify Rules number: %d\n" % QR_num)
    f.write("\nRules number: %d\n" % R_num)
    f.write("Qualify Rules number: %d\n\n" % QR_num)
    f.close()


if __name__ == '__main__':
    begin = time.time()

    print("\nThe benchmark is " + BENCHMARK + ".")
    predicateName, _ = r.RSALW.get_pre(BENCHMARK, filename='./benchmarks/')
    predicateSize = len(predicateName)
    print("Total predicates:%d" % predicateSize)
    # get ALL FACTS!
    fact_size, facts_all = r.RSALW.get_facts(BENCHMARK, filename="./benchmarks/")

    # 0:matrix 1:vector
    total_num_rule = 0
    total_time = 0

    # test_Pre_list = np.random.randint(0, predicateSize, size=5)
    test_Pre_list = [57, 163]
    # for Pt in range(predicateSize):
    for Pt in test_Pre_list:
        Pt_start = time.time()
        Pt_i_1 = Pt_start
        # Initialization all the variables.
        num_rule = 0
        ent_emb = None
        rel_emb = None
        nowPredicate = None
        fact_dic = None
        pre = None
        P_new_index_list = None

        # Firstly, sample all the elements!
        print("\n##Begin to sample##\n")
        save_path = './sampled/' + BENCHMARK
        # After sample by Pt, return the E_i-1 and F_i-1.
        # Can not remove the repeated facts!
        facts, ent_size_all = s.read_data(BENCHMARK)
        E_0, P_0, F_0, facts = s.first_sample_by_Pt(Pt, facts)
        # Initialization the sample variables.
        E = E_0
        P = P_0
        F = F_0
        E_i_1_new = E_0
        P_i_list = [P_0]
        max_i = int((Max_rule_length + 1) / 2)
        candidate_of_Pt = []
        for length in range(2, Max_rule_length+1):
            cur_max_i = int((length+1)/2)
            if len(P_i_list) < cur_max_i+1:
                # If the next P_i hasn't be computed:
                print("\nNeed to compute the P_%d\n" % cur_max_i)
                E_i, P_i, F_i_new, facts = s.sample_by_i(cur_max_i, E_i_1_new, facts)
                # Get the next cycle's variable.
                E_i_1_new = E_i - E  # remove the duplicate entity.
                # Merge the result.
                E = E | E_i
                P = P | P_i
                F.extend(F_i_new)
                P_i_list.append(P_i)
                nowPredicate, P_new_index_list = s.save_and_reindex(length, save_path, E, P, F, Pt, predicateName,
                                                                    P_i_list)
                print("\n##End to sample##\n")

                print("\nGet ALL FACTS dictionary!")
                t = time.time()
                _, pre = r.RSALW.get_pre(BENCHMARK, "./sampled/")
                fact_dic = r.RSALW.get_fact_dic(pre, facts_all, IsUncertain)
                print("Time: %s \n" % str(time.time() - t))

                print("\n##Begin to train embedding##\n")
                # The parameter of model should be adjust to the best parameters!
                ent_emb, rel_emb = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha,
                                                 lmbda, bern, margin, model)
                print("\n##End to train embedding##\n")

                # Garbage collection.
                if not gc.isenabled():
                    gc.enable()
                gc.collect()
                gc.disable()
            else:
                print("\nNeedn't to compute the next P_i")
                print("##End to sample##\n")

                print("\n##Begin to train embedding##")
                print("Needn't to train embedding")
                print("##End to train embedding##\n")

            print("\n##Begin to search and evaluate##\n")
            # Init original object
            rsalw = r.RSALW()
            candidate = rsalw.search_and_evaluate(BENCHMARK, IsUncertain, 1, length, dimension, DEGREE, nowPredicate,
                                                  ent_emb, rel_emb, ent_size_all, fact_dic,
                                                  _syn, _coocc, P_new_index_list)
            candidate_of_Pt.extend(candidate)
            print("\n##End to search and evaluate##\n")

            # Save rules and timing.
            save_rules(length, nowPredicate, candidate, pre)
            num_rule = num_rule + len(candidate)
            Pt_i = time.time()
            print("Length = %d, Time = %f" % (length, (Pt_i-Pt_i_1)))

            # Send report process E-mail!
            subject = 'ruleLearning_RainbowWu'
            text = "Pt:" + str(Pt)+ '\nLearning length of \'' + str(length) + '\' is DONE!' + '\nWe are going to start loop of \'' \
                   + str(length + 1) + '\'!\n'
            nu = "The number of rules is " + str(len(candidate)) + ".\n"
            ti = "The time of this length is " + str(Pt_i-Pt_i_1) + ".\n"
            Pt_i_1 = Pt_i
            text = BENCHMARK + ": " + text + nu + ti
            # Send email.
            send_process_report_email.send_email_main_process(subject, text)

            # Garbage collection.
            if not gc.isenabled():
                gc.enable()
            del candidate, rsalw
            gc.collect()
            gc.disable()

        total_num_rule = total_num_rule + num_rule
        Pt_end = time.time()
        Pt_time = Pt_end - Pt_start
        total_time = total_time + Pt_time
        print("This %d th predicate's total time: %f\n" % (Pt, Pt_time))
        print("Until now, all %d predicates' average time: %f\n" % (Pt, total_time/(Pt+1)))

        # Save for link prediction.
        # with open('./rule/' + BENCHMARK + '/rule_' + str(Pt) + '.pk', 'wb') as fp:
            # pickle.dump(candidate_of_Pt, fp)

    with open('./rule/'+BENCHMARK+'/rule_' + str(model)[15:21]+'.txt', 'a+') as f:
        f.write("\nEmbedding parameter:\n")
        f.write("model: %s\n" % str(model))
        f.write("train_times: %d\n" % train_times)
        f.write("dimension: %d\n" % dimension)
        f.write("alpha: %f\n" % alpha)
        f.write("lmbda: %f\n" % lmbda)
        f.write("bern: %f\n" % bern)
        f.write("\nR_minSC:%f, R_minHC:%f\n" % (R_minSC, R_minHC))
        f.write("QR_minSC:%f, QR_minHC:%f\n" % (QR_minSC, QR_minHC))
        f.write("_syn: %f\n" % _syn)
        f.write("_coocc: %f\n" % _coocc)
        f.write("Total number of rules: %d\n" % total_num_rule)
        f.write("Average time:%s\n" % str(total_time/int(predicateSize)))  # unless it runs to end.

        # Total time:
        end = time.time() - begin
        hour = int(end / 3600)
        minute = int((end - hour * 3600) / 60)
        second = end - hour * 3600 - minute * 60
        print("\nAlgorithm total time: %f" % end)
        print(str(hour) + " : " + str(minute) + " : " + str(second))

        f.write("Algorithm total time: %d : %d : %f" % (hour, minute, second))
        f.close()
