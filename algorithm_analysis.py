# -*- coding: utf-8 -*
import sampling_analysis as s
import numpy as np
import gc
import time
import send_process_report_email
import csv
import pickle
'''
import sys
sys.stdout.write('\r'+str())
sys.stdout.flush()
'''
BENCHMARK = "Wiki"
Max_rule_length = 4
# not include head atom


if __name__ == '__main__':
    begin = time.time()
    print("\nThe benchmark is " + BENCHMARK + ".")
    predicate_all, predicate_size = s.get_pre(BENCHMARK, filename='./benchmarks/')
    predicate_name = [p[0] for p in predicate_all]
    facts_all, ent_size_all = s.read_data(BENCHMARK, filename="./benchmarks/")
    # facts_all: has a flag to identify its usage.
    print("Total predicates:%d" % predicate_size)
    print("Total entities:%d" % ent_size_all)
    print("Total facts:%d" % len(facts_all))
    total_time = 0
    # test_Pre_list = np.random.randint(0, predicateSize-1, size=5)
    # test_Pre_list = [0, 3, 52, 102, 163]  # FB15k
    test_Pre_list = [15, 49, 58, 84, 135, 177, 31, 22, 220, 325, 99, 56, 187, 364, 146, 345, 180, 151, 42, 114]
    # test_Pre_list = [16, 32, 98, 314, 500, 480, 160, 45, 90, 121, 531, 285, 580, 613, 380, 289, 485, 282, 1]  # DB 19?
    # test_Pre_list = [6, 8, 13, 24, 35, 29, 32, 22, 18, 34, 16, 1, 25, 11, 0, 4, 27, 28, 30, 3]  # yago
    # test_Pre_list = [15, 49, 58, 84, 135, 177, 31, 22, 220, 325, 99, 56, 187, 364, 146, 345, 180, 151, 42, 114] # wiki
    lines = []
    plines = []
    pps = []
    i = 1
    for Pt in test_Pre_list:
        Pt_start = time.time()
        Pt_i_1 = Pt_start

        # Initialization all the variables.
        num_rule = 0
        ent_emb = None
        rel_emb = None
        new_index_Pt = None
        fact_dic_sample = None
        fact_dic_all = None
        facts_sample = None
        ent_size_sample = None
        pre_sample = None
        P_i_list_new = None
        P_count_new = None
        candidate_of_Pt = []

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()

        print(Pt)
        line = []
        pline = []
        pp=[]
        print("\n##Begin to sample##\n")
        # After sampling by Pt, return the E_0 and F_0.
        E_0, P_0, F_0, facts_all, E_0_all = s.first_sample_by_Pt(Pt, facts_all)
        line.extend([i, Pt, len(E_0), len(F_0)])
        pline.extend([i, Pt])
        pp.extend([i, Pt, ":"])
        i += 1
        # Initialization the sample variables.
        E = E_0
        P = P_0
        F = F_0
        E_i_1_new = E_0
        P_i_list = [list(P_0)]
        max_i = int((Max_rule_length + 1) / 2)
        for length in range(2, Max_rule_length+1):  # body length
            cur_max_i = int((length+1)/2)
            if len(P_i_list) < cur_max_i+1:
                # If the next P_i hasn't be computed:
                print("\nNeed to compute the P_%d." % cur_max_i)
                E_i, P_i, F_i_new, facts_all, P_count_list = s.sample_by_i(cur_max_i, E_i_1_new, facts_all)
                pline.extend([len(P_count_list), np.min(P_count_list), np.max(P_count_list), np.mean(P_count_list)])
                pp.append(P_count_list)
                # Get the next cycle's variable.
                E_i_1_new = E_i - E  # remove the duplicate entity.
                line.extend([len(E_i_1_new), len(F_i_new)])
                print("The new entity size :%d   (RLvLR need to less than 800.)" % len(E_i_1_new))
                # Merge the result.
                E = E | E_i  # set
                P = P | P_i  # set
                F.extend(F_i_new)
                P_i.add(Pt)
                P_i_list.append(list(P_i))
                print("\n##End to sample##\n")
                if not gc.isenabled():
                    gc.enable()
                gc.collect()
                gc.disable()
        line.extend([len(E), len(F), len(P)])
        lines.append(line)
        plines.append(pline)
        pps.append(pp)
        # After the mining of Pt, facts_all's usage need to be set 0.
        facts_all = np.delete(facts_all, -1, axis=1)
        fl = np.zeros(facts_all.shape[0], dtype='int32')
        facts_all = np.c_[facts_all, fl]
    with open(BENCHMARK+"_E_iAndF_i.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ent_size_all, len(facts_all), predicate_size])
        writer.writerow(["index", "Pt", "E_0", "F_0", "E_1_new", "F_1_new", "E_2_new", "F_2_new", "E", "F", "P"])
        writer.writerows(lines)
    with open(BENCHMARK + "_P_i.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "Pt", "P_1 num", "P_1 min", "P_1 max", "P_1 AVG",
                         "P_2 num", "P_2 min", "P_2 max", "P_2 AVG"])
        writer.writerows(plines)
        writer.writerows(pps)
    subject = "End of statistics!"
    text = "Let's watch the result! Go Go Go!\n"
    send_process_report_email.send_email_main_process(subject, text)
