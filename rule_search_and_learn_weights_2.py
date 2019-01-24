import time
import numpy as np
from scipy import sparse
import model_learn_weights as mlw
import itertools


class RSALW(object):
    def __int__(self):
        self.fact_dic_all = {}
        self.entity_size_all = 0

    def sim(self, para1, para2):  # similarity of vector or matrix
        return np.e ** (-np.linalg.norm(para1 - para2, ord=2))


    def index_convert(self, n, relzise, length):
        a = []
        while True:
            s = n // relzise
            y = n % relzise
            a = a + [y]
            if s == 0:
                break
            n = s
        le = length - len(a)
        if le != 0:
            for _ in range(le):
                a.append(0)
        a.reverse()
        return a


    def is_repeated(self, length, M_index):
        for i in range(1, length):
            if M_index[i] < M_index[0]:
                return True
        return False


    def scorefunction1(self, flag, syn, pt, relation, length):  # synonymy!
        relsize = relation.shape[0]
        index_list = []
        for i in range(pow(relsize, length)):
            temp = self.index_convert(i, relsize, length)
            index_list.append(temp)
        # print(index_list)
        for index in index_list:
            M_index = [[i] for i in index]
            M = [relation[i] for i in index]
            # print(M)
            # print(M_index)
            if flag == 0:  # matrix
                result = np.linalg.multi_dot(M)
            else:  # vector
                if self.is_repeated(length, M_index):
                    continue
                else:
                    result = sum(M)
            # print(syn[M_index])
            syn[M_index] = self.sim(result, relation[pt])
            # print(syn[M_index])
        print("\nf1 matrix: ")
        print(syn)

    def scorefunction2(self, coocc, relsize, facts, entity, pt, length):  # co-occurrence
        # get the different object and subject for every predicate
        objdic = {}  # key:predicate value: set
        subdic = {}  # key:predicate value: set
        factdic = {}  # key:predicate value: list
        for x in range(facts.shape[0]):
            if facts[x, 2] not in objdic:  # new key
                tempsub = set()
                tempobj = set()
                tempfact = []
            else:
                tempsub = subdic.get(facts[x, 2])
                tempobj = objdic.get(facts[x, 2])
                tempfact = factdic.get(facts[x, 2])
            tempsub.add(facts[x, 0])
            tempobj.add(facts[x, 1])
            tempfact.append(facts[x, :])
            subdic[facts[x, 2]] = tempsub
            objdic[facts[x, 2]] = tempobj
            factdic[facts[x, 2]] = tempfact
        # get the average vector of average predicate which is saved in the dictionary.
        average_vector = {}
        for key in subdic:
            # print(key)
            sub = sum([entity[item, :] for item in subdic[key]]) / len(subdic[key])
            obj = sum([entity[item, :] for item in objdic[key]]) / len(objdic[key])
            average_vector[key] = [sub, obj]
        # print("\n the dic's size is equal to the predicates' number! ")
        # print(len(average_vector))
        index_list = []
        for i in range(pow(relsize, length)):
            temp = self.index_convert(i, relsize, length)
            index_list.append(temp)
        # print(index_list)
        for index in index_list:
            M_index = [[i] for i in index]
            # print(coocc[M_index])
            para_sum = 0.0
            for i in range(length - 1):
                para_sum = para_sum + self.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
            coocc[M_index] = para_sum + self.sim(average_vector.get(index[0])[0], average_vector.get(pt)[0]) \
                             + self.sim(average_vector.get(index[length - 1])[1], average_vector.get(pt)[1])
            # print(coocc[M_index])
        print("\nf2 matrix: ")
        print(coocc)
        return factdic

    def getmatrix(self, p):
        # sparse matrix
        # print(self.fact_dic_all)
        pfacts = self.fact_dic_all.get(p)
        pmatrix = sparse.dok_matrix((self.entity_size_all, self.entity_size_all), dtype=np.int32)
        for f in pfacts:
            pmatrix[f[0], f[1]] = 1
        return pmatrix

    def calSCandHC(self, pmatrix, ptmatrix):
        head = len(ptmatrix)
        supp = 0
        body = 0
        # calculate New SC
        supp_score = 0
        body_score = 0
        for key in pmatrix.keys():
            body = body + 1
            body_score = body_score + pmatrix[key[0], key[1]]
            if ptmatrix[key[0], key[1]] == 1:
                supp = supp + 1
                supp_score = supp_score + pmatrix[key[0], key[1]]
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        if body_score == 0:
            New_SC = 0
        else:
            New_SC = supp_score / body_score
        return New_SC, SC, HC

    def evaluate_and_filter(self, pt, index, DEGREE):
        # Evaluation certain rule.
        M = [self.getmatrix(i) for i in index]
        pmatrix = sparse.dok_matrix(np.linalg.multi_dot(M))
        ptmatrix = self.getmatrix(pt)
        # calculate the SC and HC
        NSC, SC, HC = self.calSCandHC(pmatrix, ptmatrix)
        # 1: quality rule
        # 2: high quality rule
        if SC >= DEGREE[0] and HC >= DEGREE[1]:
            print("\nThis is " + str(index))
            print("The Head Coverage of this rule is " + str(HC))
            print("The Standard Confidence of this rule is " + str(SC))
            print("The NEW Standard Confidence of this rule is " + str(NSC))
            if SC >= DEGREE[2] and HC >= DEGREE[3]:
                print("WOW, a high quality rule!")
                return 2
            return 1
        return 0

    def learn_weights(self, fact_dic, candidate, entsize, pt):
        # [[37, 0], [19, 0], [59, 0], [8, 0]]
        rule_Length = 2
        training_Iteration = 50
        learning_Rate = 0.1
        regularization_rate = 0.1

        model = mlw.LearnModel()
        model.__int__(rule_Length, training_Iteration, learning_Rate, regularization_rate, fact_dic, entsize)
        model.load_data(candidate, pt)
        model.train()
        return 0

    @staticmethod
    def get_facts(BENCHMARK, filename):
        with open(filename + BENCHMARK + "/Fact.txt") as f:
            factsSize = f.readline()
            facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
        return int(factsSize), facts

    # Generally, get predicates after sampled.
    @staticmethod
    def get_pre(BENCHMARK):
        with open("./sampled/" + BENCHMARK + "/relation2id.txt") as f:
            preSize = f.readline()
            pre = [line.strip('\n').split(' ') for line in f.readlines()]
        return int(preSize), pre

    @staticmethod
    def get_fact_dic(pre_sample, facts_all):
        fact_dic = {}
        f = len(facts_all)
        p = int(len(pre_sample) / 2)
        for i in range(f):
            for j in range(p):
                if facts_all[i, 2] == int(pre_sample[2 * j][2]):
                    if int(pre_sample[2 * j][0]) in fact_dic.keys():
                        temp_list1 = fact_dic.get(int(pre_sample[2 * j][0]))
                        temp_list2 = fact_dic.get(int(pre_sample[2 * j + 1][0]))
                    else:
                        temp_list1 = []
                        temp_list2 = []
                    temp_list1.append([facts_all[i, 0], facts_all[i, 1]])
                    temp_list2.append([facts_all[i, 1], facts_all[i, 0]])
                    fact_dic[int(pre_sample[2 * j][0])] = temp_list1
                    fact_dic[int(pre_sample[2 * j + 1][0])] = temp_list2
        # print(fact_dic.keys())
        return fact_dic

    def search_and_evaluate(self, f, length, BENCHMARK, nowPredicate, ent_emb, rel_emb, dimension, ent_size_all, fact_dic, DEGREE):
        self.fact_dic_all = fact_dic
        self.entity_size_all = ent_size_all
        relsize = rel_emb.shape[0]
        if f == 0:
            rel_emb = np.reshape(rel_emb, [relsize, dimension, dimension])
        # print(relation.shape)  # (-1, 100, 100) or (-1, 100)
        # print(entity.shape)  # (-1, 100)

        # Score Function
        # The array's shape is decided by the length of rule.
        shape = []
        for i in range(length):
            shape.append(relsize)
        print("The shape of Matrix is %s." % str(shape))
        syn = np.zeros(shape=shape)
        coocc = np.zeros(shape=shape)
        mark_Matrix = np.zeros(shape=shape)

        # calculate the f1
        print("\nBegin to calculate the f1: synonymy")
        self.scorefunction1(f, syn, nowPredicate[0], rel_emb, length)
        # calculate the f2
        print("\nBegin to calculate the f2: Co-occurrence")
        factsSize, facts = self.get_facts(BENCHMARK, filename="./sampled/")
        # print(facts)
        _fact_dic = self.scorefunction2(coocc, relsize, facts, ent_emb, nowPredicate[0], length)

        # How to choose this value to get candidate rules? Important!
        candidate = []
        print("Begin to get candidate rules.")
        # Method 1: Top ones until it reaches the 100th. OMIT!
        # Method 2: Use two matrices to catch rules.

        middle_syn = (np.max(syn) - np.min(syn)) * 0.6 + np.min(syn)
        rawrulelist = np.argwhere(syn > middle_syn)
        print(" Begin to use syn to filter: %d" % len(rawrulelist))
        for index in rawrulelist:
            if f == 0:  # matrix
                result = self.evaluate_and_filter(nowPredicate[0], index, DEGREE)
                candidate.append([index, result])
                mark_Matrix[index.reshape(length, 1).tolist()] = 1
            elif f == 1:  # vector
                # It needs to evaluate for all arranges of index.
                for i in itertools.permutations(index.tolist(), length):
                    # Deduplicate.
                    if mark_Matrix[np.array(i).reshape(length, 1).tolist()] == 1:
                        continue
                    _index = np.array(i)
                    result = self.evaluate_and_filter(nowPredicate[0], _index, DEGREE)
                    candidate.append([_index, result])
                    mark_Matrix[np.array(i).reshape(length, 1).tolist()] = 1

        middle_coocc = (np.max(coocc) - np.min(coocc)) * 0.82 + np.min(coocc)
        rawrulelist = np.argwhere(coocc > middle_coocc)
        print(" Begin to use coocc to filter: %d" % len(rawrulelist))
        for index in rawrulelist:
            if mark_Matrix[index.reshape(length, 1).tolist()] == 0:
                result = self.evaluate_and_filter(nowPredicate[0], index, DEGREE)
                if result != 0:
                    candidate.append([index, result])

        # Evaluation is still a cue method!

        print("\n*^_^* Yeah, there are %d rules. *^_^*\n" % len(candidate))
        # learn_weights(fact_dic, candidate, entsize, nowPredicate[0])  #ent_size_all??? or entsize.
        return candidate
