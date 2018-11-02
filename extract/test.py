import numpy as np
import pickle

BENCHMARK = "FB15K-237"
entity_size = 14541
relation_size = 237

ent_dic = {}
rel_dic = {}
fact_list = []
with open("./train.txt", 'r') as f:
    fact_name = [line.strip("\n").split("\t") for line in f.readlines()]
    f = np.array(fact_name)
    # print(facts)
    for fact in f:
        if fact[1] not in rel_dic.keys():
            rel_dic[fact[1]] = len(rel_dic)
        r = rel_dic[fact[1]]
        if fact[0] not in ent_dic.keys():
            ent_dic[fact[0]] = len(ent_dic)
        e1 = ent_dic[fact[0]]
        if fact[2] not in ent_dic.keys():
            ent_dic[fact[2]] = len(ent_dic)
        e2 = ent_dic[fact[2]]
        fact_list.append([e1, e2, r])

with open("./test.txt", 'r') as f:
    fact_name = [line.strip("\n").split("\t") for line in f.readlines()]
    f = np.array(fact_name)
    # print(facts)
    for fact in f:
        if fact[1] not in rel_dic.keys():
            rel_dic[fact[1]] = len(rel_dic)
        r = rel_dic[fact[1]]
        if fact[0] not in ent_dic.keys():
            ent_dic[fact[0]] = len(ent_dic)
        e1 = ent_dic[fact[0]]
        if fact[2] not in ent_dic.keys():
            ent_dic[fact[2]] = len(ent_dic)
        e2 = ent_dic[fact[2]]
        fact_list.append([e1, e2, r])

with open("./valid.txt", 'r') as f:
    fact_name = [line.strip("\n").split("\t") for line in f.readlines()]
    f = np.array(fact_name)
    # print(facts)
    for fact in f:
        if fact[1] not in rel_dic.keys():
            rel_dic[fact[1]] = len(rel_dic)
        r = rel_dic[fact[1]]
        if fact[0] not in ent_dic.keys():
            ent_dic[fact[0]] = len(ent_dic)
        e1 = ent_dic[fact[0]]
        if fact[2] not in ent_dic.keys():
            ent_dic[fact[2]] = len(ent_dic)
        e2 = ent_dic[fact[2]]
        fact_list.append([e1, e2, r])
    # facts = np.array(fact_list)
    # print(np.max(fact[:, 2]))
    # print(len(ent_dic))
    # print(len(rel_dic))
    # print(len(facts))

# save to facts
f = open('./Fact.txt', 'w')
f.write(str(len(fact_list)) + "\n")
for line in fact_list:
    f.write(str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")
f.close()

# entity
f = open('Entity.txt', 'w')
f.write(str(len(ent_dic)) + "\n")
for key_name in ent_dic.keys():
    f.write(str(ent_dic[key_name]) + str(key_name) + "\n")

# predicate
pred = []
for key_name in rel_dic.keys():
    pred.append([rel_dic[key_name], key_name])
with open('Relation', 'wb') as f:
    pickle.dump(pred, f)
