# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:36:41 2018

@author: Rainbow
"""
import numpy as np

# key:name  value:index
ent_dic = {}
rel_dic = {}
train_fact_list = []
test_fact_list = []
def read(filename, ent_dic, rel_dic, fact_list):
    with open(filename, 'r') as f:
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

read("train.txt",ent_dic, rel_dic, train_fact_list)
read("valid.txt", ent_dic, rel_dic, train_fact_list)
read("test.txt", ent_dic, rel_dic, test_fact_list)
train_fact = np.array(train_fact_list, dtype=np.int32)
test_fact = np.array(test_fact_list, dtype=np.int32)

# Save in files.
with open('./train/Fact.txt', 'w') as f:
    f.write(str(len(train_fact)) + '\n')
    for fact in train_fact:
        f.write(str(fact[0]) + ' ' + str(fact[1]) + ' ' + str(fact[2]) + '\n')
with open('./test/Fact.txt', 'w') as f:
    f.write(str(len(test_fact)) + '\n')
    for fact in test_fact:
        f.write(str(fact[0]) + ' ' + str(fact[1]) + ' ' + str(fact[2]) + '\n')

# test: 224 relations' id.
test_pre = np.unique(test_fact[:, 2])
print('test predicate num: '+str(len(test_pre)))
with open('./test/target_pre.txt', 'w') as f:
    f.write(str(len(test_pre)) + '\n')
    for p in test_pre:
        f.write(str(p) + '\n')
print("Over!")

# print(len(fact_list))
# a = list(fact_list[:, 0])
# b = list(fact_list[:, 1])
# a.extend(b)
# print(len( np.unique( a )))
# print( len( np.unique( train )) )

#Entity
# train    14505
# test     10348
# valid    9809

#Relation
# train    237 
# test     224
# valid    223