import time
import sampling_3 as s
from models import TransE, TransD, TransH, TransR, RESCAL
import train_embedding as te
import rule_search_and_learn_weights as rsalw

'''
import sys
sys.stdout.write("")
sys.flush()
'''

BENCHMARK = "FB15K237"
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
train_times = 100  # 1000
dimension = 50  # 50
alpha = 0.01  # learning rate
lmbda = 0.1  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
model = TransE.TransE
# Vetor:DistMult.DistMult HolE.HolE
#       TransE.TransE
# Unknown: TransD.TransD TransH.TransH TransR.TransR
# Matrix: RESCAL.RESCAL

# not changed yet.
def save_rules(nowPredicate, candidate, pre):
    print("\nThe final rules are:")
    i = 1
    f = open('./rule/' + BENCHMARK + '/rule_' + str(model)[15:21] + '.txt', 'a+')
    print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    f.write("num: %d\n" % len(candidate))
    R_num = 0
    QR_num = 0
    for rule in candidate:
        title = ""
        if rule[1] == 1:
            R_num = R_num + 1
            title = "Rule "
        elif rule[1] == 2:
            QR_num = QR_num + 1
            title = "Qualify Rule "
        line = title + str(i) + ": " + str(rule[0][0]) + " " + pre[rule[0][0]][1] + "  &&  " \
               + str(rule[0][1]) + " " + pre[rule[0][1]][1] + "\n"
        print(line)
        f.write(line)
        i = i + 1
    f.write("Rules number: %d\n" % R_num)
    f.write("Qualify Rules number: %d\n" % QR_num)
    f.close()


if __name__ == '__main__':
    begin = time.time()
    print("\nThe benchmark is " + BENCHMARK + ".\n")
    with open('./benchmarks/' + BENCHMARK + '/relation2id.txt', 'r') as f:
        predicateSize = int(f.readline())
        predicateName = [relation.split("	")[0] for relation in f.readlines()]
        print("Total predicates:" + str(predicateSize))
    num_rule = 0
    total_time = 0
    # 0:matrix 1:vector
    for Pt in range(predicateSize):
        # 对于每个规则长度进行循环!
        Pt_0 = time.time()
        print("##Begin to sample##")
        s.first_sample_by_Pt(BENCHMARK, Pt, predicateName)
        # after sample by Pt, return the E_i and F_i.
        # return E_new 
        for i in range(Max_rule_length):
            
            ent_size_all, nowPredicate = s.sample(BENCHMARK, Pt, predicateName)
            # The parameter of model should be adjust to the best parameters!!!!!!
            ent_emb, rel_emb = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern,
                                             margin, model)
            _p, pre = rsalw.get_pre(BENCHMARK)
            candidate = rsalw.searchAndEvaluate(1, BENCHMARK, nowPredicate, ent_emb, rel_emb,
                                                dimension, ent_size_all, pre, DEGREE)
            save_rules(nowPredicate, candidate, pre)
            num_rule = num_rule + len(candidate)
        
        Pt_1 = time.time()
        Pt_time = Pt_1 - Pt_0
        print("Predicate target's time: " + str(Pt_time) + "\n")
        total_time = total_time + Pt_time
        print("Average's time: %f\n" % (total_time / (Pt + 1)))
        break

    f = open('./rule/' + BENCHMARK + '/rule_After_' + str(model)[15:21] + '.txt', 'a+')
    f.write("Embedding parameter:\n")
    f.write("model: %s\n" + str(model))
    f.write("train_times: %d\n" % train_times)
    f.write("dimension: %d\n" % dimension)
    f.write("alpha: %f\n" % alpha)
    f.write("lmbda: %f\n" % lmbda)
    f.write("bern: %f\n" % bern)
    f.write("\nR_minSC:%f, R_minHC:%f\n" % (R_minSC, R_minHC))
    f.write("QR_minSC:%f, QR_minHC:%f\n" % (QR_minSC, QR_minHC))
    # f.write("syn: " + str(times_syn) + "\n")
    # f.write("coocc: " + str(times_coocc) + "\n")
    f.write("Total number of rules: %d\n" % num_rule)
    f.write("Average time:%s\n" % str(total_time/int(predicateSize)))
    f.close()
    end = time.time()-begin
    hour = int(end/3600)
    minute = int((end-hour*3600)/60)
    second = end-hour*3600-minute*60
    print("\nAlgorithm total time: ", end)
    print(str(hour)+" : "+str(minute)+" : "+str(second))
