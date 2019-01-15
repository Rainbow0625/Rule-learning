import time
import sampling as s
from models import TransE, RESCAL
import train_embedding as te
import rule_search_and_learn_weights as rsalw

BENCHMARK = "FB15K237"
minSC = 0.01  # ?
minHC = 0.001  # ?

# embedding model parameters    should be modify!!!!!!!!!
work_threads = 5
nbatches = 100
margin = 1  # the margin for the loss function

train_times = 1000  # 150
dimension = 50
alpha = 0.01  # learning rate
lmbda = 0.01  # degree of the regularization on the parameters
bern = 1  # set negative sampling algorithms, unif(0) or bern(1)
model = TransE.TransE  # DistMult.DistMult   TransE.TransE   HolE.HolE  RESCAL.RESCAL

begin = time.time()
print("\nThe benchmark is " + BENCHMARK + ".\n")
with open('./benchmarks/' + BENCHMARK + '/relation2id.txt', 'r') as f:
    # list: eg: "/location/country/form_of_government	0"
    predicateSize = int(f.readline())
    predicateName = [relation.split("	")[0] for relation in f.readlines()]
    # print(predicateName)
    print("Total predicates:" + str(predicateSize))
num_rule = 0
total_time = 0
# 0:matrix 1:vector
for Pt in range(predicateSize):
    Pt_0 = time.time()
    ent_size_all, nowPredicate = s.sample0(BENCHMARK, Pt, predicateName)
    # The parameter of model should be adjust to the best parameters!!!!!!
    ent_emb, rel_emb = te.trainModel(1, BENCHMARK, work_threads, train_times, nbatches, dimension, alpha, lmbda, bern,
                                     margin, model)
    rule_of_Pt = rsalw.searchAndEvaluate(1, BENCHMARK, nowPredicate, ent_emb, rel_emb, dimension, model, ent_size_all)
    # save weights

    num_rule = num_rule + rule_of_Pt
    Pt_1 = time.time()
    Pt_time = Pt_1 - Pt_0
    print("Predicate target's time: " + str(Pt_time) + "\n")
    total_time = total_time + Pt_time
    print("Average's time: " + str(total_time / (Pt + 1)) + "\n")
    break
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
# f.write("syn: " + str(times_syn) + "\n")
# f.write("coocc: " + str(times_coocc) + "\n")
f.write("Total number of rules: " + str(num_rule) + "\n")
f.write("Average time:" + str(total_time/int(predicateSize)) + "\n")
f.write("\n")
f.close()
end = time.time()-begin
hour = int(end/3600)
minute = int((end-hour*3600)/60)
second = end-hour*3600-minute*60
print("\nAlgorithm total time: ", end)
print(str(hour)+" : "+str(minute)+" : "+str(second))
