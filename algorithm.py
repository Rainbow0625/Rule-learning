import sampling as s
import sampling_after as sa
import train_embedding as te
import rule_search as rs
import time
from models import TransE, DistMult, HolE
import pickle

# 1: "DB"  "Wiki"  "Yago"
# 0: "FB15K"  "WN18"  "FB15K237"
BENCHMARK = "NELL"
minSC = 0.01
minHC = 0.001
times_syn = 10
times_coocc = 10

# embedding model parameters
work_threads = 8
nbatches = 100
margin = 1.0  # the margin for the loss function

train_times = 150  # 150
dimension = 50
alpha = 0.01  # learning rate
lmbda = 0.01  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
model = TransE.TransE  # DistMult.DistMult   TransE.TransE   HolE.HolE

begin = time.time()
print("\nThe benchmark is " + BENCHMARK + ".\n")


def save_rules(BENCHMARK, candidate):
    with open("./sampled/" + BENCHMARK + "/relation2id.txt") as f:
        preSize = f.readline()
        pre = [line.strip('\n').split(' ') for line in f.readlines()]
    print("\nThe final rules are:")
    i = 1
    f = open('./rule/' + BENCHMARK + '/rule_After_' + str(model)[15:21] + '.txt', 'a+')
    print(str(nowPredicate[1]) + "\n")
    f.write(str(nowPredicate[1]) + "\n")
    rule_of_Pt = len(candidate)
    print(str(rule_of_Pt) + "\n")
    f.write("num: " + str(rule_of_Pt) + "\n")
    for rule in candidate:
        print(rule[0])
        print(rule[1])
        line = "Rule " + str(i) + ": " + pre[rule[0]][1] + "  &&  " + pre[rule[1]][1] + "\n"
        print(line)
        f.write(line)
        i = i + 1
    f.write("\n")
    f.close()
    return rule_of_Pt


# FB15K237  # NELL : first sampling, then embedding.
with open('./benchmarks/' + BENCHMARK + '/relation2id.txt', 'r') as f:
    # list: eg: "/location/country/form_of_government	0"
    predicateSize = int(f.readline())
    predicateName = [relation.split("	")[0] for relation in f.readlines()]
    # print(predicateName)
    print("Total predicates:" + str(predicateSize))
num_rule = 0
total_time = 0
for Pt in range(predicateSize):
    Pt_0 = time.time()
    nowPredicate = s.sample0(BENCHMARK, Pt, predicateName)
    entity, relation = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
    # entity, relation = rs.get_embedding(1, BENCHMARK)  
    candidate = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    rule_of_Pt = save_rules(BENCHMARK, candidate)
    num_rule = num_rule + rule_of_Pt
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")


f = open('./rule/' + BENCHMARK + '/rule_After_' + str(model)[15:21] + '.txt', 'a+')
f.write("Embedding parameter:" + "\n")
f.write("model: " + str(model) + "\n")
f.write("train_times: " + str(train_times) + "\n")
f.write("dimension: " + str(dimension) + "\n")
f.write("alpha: " + str(alpha) + "\n")
f.write("lmbda: " + str(lmbda) + "\n")
f.write("bern: " + str(bern) + "\n")

f.write("\n" + "minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Average time:" + str(total_time/int(predicateSize)) + "\n")
f.write("\n")
f.close()


'''
# FB15K237: first embedding, then sampling.
with open('./benchmarks/' + BENCHMARK + '/relation2id', 'r') as relationfile:
    # list: eg: [3, '/location/location/contains']
    predicateName = pickle.load(relationfile)
    predicateName = [relation[1] for relation in predicateName]
    predicateSize = len(predicateName)
    print("Total predicates:" + str(predicateSize))
num_rule = 0
total_time = 0
ent, rel = te.trainModel(0, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
# ent, rel = rs.get_embedding(0, BENCHMARK)
time_embedding = time.time()-begin
index_flag = 0
for Pt in range(int(predicateSize)):
    Pt_0 = time.time()
    entity, relation, nowPredicate = sa.sample(index_flag, BENCHMARK, Pt, predicateName, ent, rel)
    n_rule = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    num_rule = num_rule + n_rule
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")
f = open('./rule/' + BENCHMARK + '/ruleBefore.txt', 'a+')
f.write("minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Embedding time:" + str(time_embedding) + "\n")
f.write("Average Pt time:" + str(total_time/int(predicateSize)) + "\n")
f.close()
'''

# Firstly, sampling, then embedding.
'''
# 1: "DB"  "Wiki"  "Yago"
with open('./benchmarks/' + BENCHMARK + '/predindex.txt', 'r') as filePredicate:
    predicateName = [line.strip('\n').strip('[').strip(']').split(', ')[1].strip('\'') for line in filePredicate.readlines()]
    # print(predicateName)   # ['<dbo:author>','<....>']
predicateSize = len(predicateName)
num_rule = 0
total_time = 0
for Pt in range(predicateSize):
    Pt_0 = time.time()
    nowPredicate = s.sample1(BENCHMARK, Pt, predicateName)
    entity, relation  = trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
    # entity, relation = rs.get_embedding(1, BENCHMARK)
    n_rule = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    num_rule = num_rule + n_rule
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: "+str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time/(Pt+1)) + "\n")
f = open('./rule/' + BENCHMARK + '/ruleAfter.txt', 'a+')
f.write("minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Average time:" + str(total_time/int(predicateSize)) + "\n")
f.close()
'''

'''
# 0: "FB15K"  "WN18"
with open('./benchmarks/' + BENCHMARK + '/Relation.txt', 'r') as relationfile:
    predicateSize = relationfile.readline()
    print("Total predicates:" + str(predicateSize))
    predicateName = [line.strip('\n').split('\t')[0] for line in relationfile.readlines()]
num_rule = 0
total_time = 0
for Pt in range(int(predicateSize)):
    Pt_0 = time.time()
    nowPredicate = s.sample0(BENCHMARK, Pt, predicateName)
    entity, relation = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
    # entity, relation = rs.get_embedding(1, BENCHMARK)  
    n_rule = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    num_rule = num_rule + n_rule
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")
f = open('./rule/' + BENCHMARK + '/ruleAfter.txt', 'a+')
f.write("minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Average time:" + str(total_time/int(predicateSize)) + "\n")
f.close()
'''

'''
# Firstly, embedding, then sampling.
# 0: "FB15K"  "WN18"
with open('./benchmarks/' + BENCHMARK + '/Relation.txt', 'r') as relationfile:
    predicateSize = relationfile.readline()
    print("Total predicates:"+ str(predicateSize))
    predicateName = [line.strip('\n').split('\t')[0] for line in relationfile.readlines()]
num_rule = 0
total_time = 0
ent, rel = te.trainModel(0, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
# ent, rel = rs.get_embedding(0, BENCHMARK)
time_embedding = time.time()-begin
index_flag = 0
for Pt in range(int(predicateSize)):
    Pt_0 = time.time()
    entity, relation, nowPredicate = sa.sample(index_flag, BENCHMARK, Pt, predicateName, ent, rel)
    n_rule = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    num_rule = num_rule + n_rule
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")
f = open('./rule/' + BENCHMARK + '/ruleBefore.txt', 'a+')
f.write("minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Embedding time:" + str(time_embedding) + "\n")
f.write("Average Pt time:" + str(total_time/int(predicateSize)) + "\n")
f.close()
'''

'''
# 1: "DB"  "Wiki"  "Yago"
sa.preprocess(BENCHMARK)  # pre process to DB, Wiki and Yago
with open('./benchmarks/' + BENCHMARK + '/Relation.txt', 'r') as relationfile:
    predicateSize = relationfile.readline()
    print("Total predicates:" + str(predicateSize))
    predicateName = [line.strip('\n').split('\t')[0] for line in relationfile.readlines()]
num_rule = 0
total_time = 0
ent, rel = te.trainModel(0, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern, margin, model)
# ent, rel = rs.get_embedding(0, BENCHMARK)
time_embedding = time.time()-begin
print("Embedding time:" + str(time_embedding)+"\n")
index_flag = 1
for Pt in range(int(predicateSize)):
    Pt_0 = time.time()
    entity, relation, nowPredicate = sa.sample(index_flag, BENCHMARK, Pt, predicateName, ent, rel)
    n_rule = rs.searchAndEvaluate(BENCHMARK, nowPredicate, minSC, minHC, times_syn, times_coocc, entity, relation)
    num_rule = num_rule + n_rule
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")
f = open('./rule/' + BENCHMARK + '/ruleBefore.txt', 'a+')
f.write("minSC: " + str(minSC) + "\n")
f.write("minHC: " + str(minHC) + "\n")
f.write("syn: " + str(times_syn) + "\n")
f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule)+ "\n")
f.write("Embedding time:" + str(time_embedding)+ "\n")
f.write("Average Pt time:" + str(total_time/int(predicateSize))+ "\n")
f.close()
'''

end = time.time()-begin
hour = int(end/3600)
minute = int((end-hour*3600)/60)
second = end-hour*3600-minute*60
print("\nAlgorithm total time: ", end)
print(str(hour)+" : "+str(minute)+" : "+str(second))
