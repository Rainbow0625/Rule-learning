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
Max_rule_length = 4  # not include head atom
_syn = 800
_coocc = 800

# embedding model parameters
work_threads = 5
nbatches = 150
margin = 1  # the margin for the loss function
train_times = 10  # 1000
dimension = 50  # 50
alpha = 0.01  # learning rate
lmbda = 0.01  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
model = TransE.TransE
# Vetor:DistMult.DistMult HolE.HolE
#       TransE.TransE
# Unknown: TransD.TransD TransH.TransH TransR.TransR
# Matrix: RESCAL.RESCAL


def save_rules(Pt, rule_length, nowPredicate, candidate, pre):
    print("The final rules :")
    # str(model)[15:21]
    f = open('./rule/' + BENCHMARK + '/rule_' + str(Pt) + '.txt', 'a+')
    # print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    f.write("length: %d, num: %d\n" % (rule_length, len(candidate)))
    R_num = 0
    QR_num = 0
    i = 1
    rule_aed_list = []
    HC_value_list = []
    for rule in candidate:
        index = rule[0]
        flag = rule[1]
        degree = str(rule[2])
        # Duplicate elimination.
        if rule[2][1] not in HC_value_list:
            rule_aed_list.append(rule)
            HC_value_list.append(rule[2][1])

        # Save Quality rules and rules.
        if flag == 1:
            R_num = R_num + 1
            title = "Rule " + str(i) + ": "
        else:
            QR_num = QR_num + 1
            title = "Qualify Rule " + str(i) + ": "
        line = title + " " + str(index) + " :[SC, HC] " + degree + " "
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

    with open('./rule/' + BENCHMARK + '/rule_aed' + str(Pt) + '.txt', 'a+') as fp:
        i = 0
        for rule in rule_aed_list:
            index = rule[0]
            flag = rule[1]
            degree = str(rule[2])
            # Save Quality rules and rules.
            if flag == 1:
                R_num = R_num + 1
                title = "Rule " + str(i) + ": "
            else:
                QR_num = QR_num + 1
                title = "Qualify Rule " + str(i) + ": "
            line = title + " " + str(index) + " :[SC, HC] " + degree + " "
            for j in range(rule_length):
                line = line + str(index[j]) + " " + pre[index[j]][1] + "; "
            line = line + "\n"
            # print(line)
            fp.write(line)
            i = i + 1


if __name__ == '__main__':
    begin = time.time()

    print("\nThe benchmark is " + BENCHMARK + ".")
    predicateName, _ = r.RSALW.get_pre(BENCHMARK, filename='./benchmarks/')
    predicateSize = len(predicateName)
    print("Total predicates:%d" % predicateSize)
    facts_all, ent_size_all = s.read_data(BENCHMARK, filename="./benchmarks/")
    # 0:matrix 1:vector
    total_num_rule = 0
    total_time = 0

    # test_Pre_list = np.random.randint(0, predicateSize-1, size=5)
    # test_Pre_list = [0, 3, 52, 102, 163, 12, 27, 47]
    test_Pre_list = [0, 3, 52, 102, 163, 12, 27, 47]
    # for Pt in range(predicateSize):
    for Pt in test_Pre_list:
        Pt_start = time.time()
        Pt_i_1 = Pt_start
        # Initialization all the variables.
        num_rule = 0
        ent_emb = None
        rel_emb = None
        nowPredicate = None
        fact_dic_sample = None
        fact_dic_all = None
        facts_sample = None
        ent_size_sample = None
        pre_sample = None
        P_new_index_list = None
        P_count_dic = None

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()

        # Firstly, sample all the elements!
        print("\n##Begin to sample##\n")
        save_path = './sampled/' + BENCHMARK
        # After sample by Pt, return the E_0 and F_0.
        E_0, P_0, F_0, facts_all = s.first_sample_by_Pt(Pt, facts_all)
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
                E_i, P_i, F_i_new, facts_all, _P_count = s.sample_by_i(cur_max_i, E_i_1_new, facts_all)
                # Get the next cycle's variable.
                E_i_1_new = E_i - E  # remove the duplicate entity.
                print("The new entity size :%d   need to less than 800?" % len(E_i_1_new))
                # Merge the result.
                E = E | E_i
                P = P | P_i
                F.extend(F_i_new)
                P_i_list.append(P_i)
                # _P_count list are old indices; P_count dictionary's keys are new indices.
                nowPredicate, P_new_index_list, P_count_dic = s.save_and_reindex(length, save_path, E, P, F, Pt,
                                                                                 predicateName, P_i_list, _P_count)
                print("\n##End to sample##\n")

                print("\nGet SAMPLE PREDICATE dictionary. (First evaluate on small sample KG.)")
                t = time.time()
                # The predicates in "pre_sample" is the total number written in file.
                _, pre_sample = r.RSALW.get_pre(BENCHMARK, "./sampled/")
                facts_sample, ent_size_sample = s.read_data(BENCHMARK, filename="./sampled/")
                # First get fact_dic_sample.
                fact_dic_sample = r.RSALW.get_fact_dic_sample(facts_sample)
                fact_dic_all = r.RSALW.get_fact_dic_all(pre_sample, facts_all)
                print("Time: %s \n" % str(time.time() - t))

                print("\n##Begin to train embedding##\n")
                # The parameter of model should be adjust to the best parameters!
                ent_emb, rel_emb = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha,
                                                 lmbda, bern, margin, model)
                print("\n##End to train embedding##\n")

                isfullKG = True

                # Garbage collection.
                if not gc.isenabled():
                    gc.enable()
                gc.collect()
                gc.disable()
            else:
                print("\nNeedn't to compute the next P_i")
                print("Filter out predicates that appear too frequently to reduce the computational time complexity.\n")
                P_new_index_list = s.filter_predicates_by_count(P_count_dic, P_new_index_list)
                print("After filter, the length of pre: %d :%d " % (len(P_new_index_list[-1]), len(P_count_dic)))
                print("##End to sample##")

                print("\n##Begin to train embedding##")
                print("Needn't to train embedding")
                print("##End to train embedding##\n")

                isfullKG = False

                # Garbage collection.
                if not gc.isenabled():
                    gc.enable()
                del P_count_dic
                gc.collect()
                gc.disable()
            print("\n##Begin to search and evaluate##\n")
            # Init original object
            rsalw = r.RSALW()
            print(len(fact_dic_sample))
            print(len(fact_dic_all))
            candidate = rsalw.search_and_evaluate(IsUncertain, 1, length, dimension, DEGREE, nowPredicate,
                                                  ent_emb, rel_emb, _syn, _coocc, P_new_index_list, isfullKG,
                                                  fact_dic_sample, fact_dic_all, ent_size_sample, ent_size_all)
            candidate_of_Pt.extend(candidate)
            candidate_len = len(candidate)
            num_rule = num_rule + candidate_len
            print("\n##End to search and evaluate##\n")

            # Save rules and timing.
            save_rules(Pt, length, nowPredicate, candidate, pre_sample)
            Pt_i = time.time()
            print("Length = %d, Time = %f" % (length, (Pt_i-Pt_i_1)))

            # Garbage collection.
            if not gc.isenabled():
                gc.enable()
            del candidate, rsalw
            gc.collect()
            gc.disable()

            # Send report process E-mail!
            subject = 'ruleLearning'
            text = "Pt:" + str(Pt) + '\nLength: ' + str(length) + '\n'
            nu = "The number of rules: " + str(candidate_len) + "\n"
            ti = "The time of this length: " + str(Pt_i-Pt_i_1)[0:5] + "\n"
            Pt_i_1 = Pt_i
            text = BENCHMARK + ": " + text + nu + ti
            # Send email.
            send_process_report_email.send_email_main_process(subject, text)

        total_num_rule = total_num_rule + num_rule
        Pt_end = time.time()
        Pt_time = Pt_end - Pt_start
        total_time = total_time + Pt_time
        print("This %d th predicate's total time: %f\n" % (Pt, Pt_time))
        print("Until now, all %d predicates' average time: %f\n" % (Pt, total_time/(Pt+1)))

        # Save for link prediction.
        # with open('./rule/' + BENCHMARK + '/rule_' + str(Pt) + '.pk', 'wb') as fp:
            # pickle.dump(candidate_of_Pt, fp)

        # After the mining of Pt, facts_all's usage need to be set 0.
        facts_all = np.delete(facts_all, -1, axis=1)
        fl = np.zeros(facts_all.shape[0], dtype='int32')
        facts_all = np.c_[facts_all, fl]

    with open('./rule/'+BENCHMARK+'/rule_top' + str(_coocc) + '_maxlen' + str(Max_rule_length) + '.txt', 'a+') as f:
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

        f.write("Algorithm total time: %d : %d : %f\n" % (hour, minute, second))

    subject = "Over!"
    text = "Let's watch the result! Go Go Go!\n"
    send_process_report_email.send_email_main_process(subject, text)
