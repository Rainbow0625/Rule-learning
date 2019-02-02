import time
import numpy as np
from scipy import sparse
import model_learn_weights as mlw
import itertools
import gc


class RSALW(object):
    def __int__(self):
        self.fact_dic_all = {}
        self.entity_size_all = 0
        self.length = 0
        self.isUncertian = False

    @staticmethod
    def sim(para1, para2):  # similarity of vector or matrix
        return np.e ** (-np.linalg.norm(para1 - para2, ord=2))

    def index_convert(self, n, relzise):
        a = []
        while True:
            s = n // relzise
            y = n % relzise
            a = a + [y]
            if s == 0:
                break
            n = s
        le = self.length - len(a)
        if le != 0:
            for _ in range(le):
                a.append(0)
        a.reverse()
        return a

    def is_repeated(self, M_index):
        for i in range(1, self.length):
            if M_index[i] < M_index[0]:
                return True
        return False

    def score_function1(self, flag, score_top_container, relation):  # synonymy!
        relsize = relation.shape[0]
        index_list = []
        for i in range(pow(relsize, self.length)):
            temp = self.index_convert(i, relsize)
            index_list.append(temp)
        # print(index_list)
        for index in index_list:
            M = [relation[i] for i in index]
            # print(M)
            if flag == 0:  # matrix
                # array
                result = np.linalg.multi_dot(M)
            else:  # vector
                if self.is_repeated(index):
                    continue
                else:
                    result = sum(M)
            top_values = score_top_container[:, self.length]
            value = self.sim(result, relation[self.pt])
            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                for i in range(self.length):
                    score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
                # print(score_top_container[replace_index])

    def score_function2(self, score_top_container, relsize, facts, entity):  # co-occurrence
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
        for i in range(pow(relsize, self.length)):
            temp = self.index_convert(i, relsize)
            index_list.append(temp)
        # print(index_list)
        for index in index_list:
            para_sum = 0.0
            for i in range(self.length - 1):
                para_sum = para_sum + self.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
            value = para_sum + self.sim(average_vector.get(index[0])[0],
                                                        average_vector.get(self.pt)[0]) \
                                    + self.sim(average_vector.get(index[self.length - 1])[1],
                                               average_vector.get(self.pt)[1])
            top_values = score_top_container[:, self.length]
            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                for i in range(self.length):
                    score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
                # print(score_top_container[replace_index])

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
        supp_score = 0.0
        body_score = 0.0
        for key in pmatrix.keys():
            body = body + 1
            body_score = body_score + pmatrix[key]
            if ptmatrix[key] == 1:
                supp = supp + 1
                supp_score = supp_score + pmatrix[key]
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        if body_score == 0.0:
            New_SC = 0
        else:
            New_SC = supp_score / body_score
        return New_SC, SC, HC

    def evaluate_and_filter(self, index, DEGREE):
        # Evaluation certain rule.
        # sparse matrix
        pmatrix = self.getmatrix(index[0])
        for i in range(1, self.length):
            pmatrix = pmatrix.dot(self.getmatrix(index[i]))
        pmatrix = pmatrix.todok()
        ptmatrix = self.getmatrix(self.pt)
        # calculate the SC and HC
        NSC, SC, HC = self.calSCandHC(pmatrix, ptmatrix)
        degree = [NSC, SC, HC]
        # 1: quality rule
        # 2: high quality rule
        if SC >= DEGREE[0] and HC >= DEGREE[1]:
            print("\nThis is " + str(index))
            print("The Head Coverage of this rule is " + str(HC))
            print("The Standard Confidence of this rule is " + str(SC))
            print("The NEW Standard Confidence of this rule is " + str(NSC))
            if SC >= DEGREE[2] and HC >= DEGREE[3]:
                print("WOW, a high quality rule!")
                return 2, degree
            return 1, degree
        return 0, None

    def learn_weights(self, candidate):
        # In the whole data set to learn the weights.
        training_Iteration = 100
        learning_Rate = 0.1
        regularization_rate = 0.1

        model = mlw.LearnModel()
        model.__int__(self.length, training_Iteration, learning_Rate, regularization_rate,
                      self.fact_dic_all, self.entity_size_all, candidate, self.pt, self.isUncertian)
        model.train()

    @staticmethod
    def get_facts(BENCHMARK, filename):
        with open(filename + BENCHMARK + "/Fact.txt") as f:
            factsSize = f.readline()
            facts = np.array([line.strip('\n').split(' ') for line in f.readlines()], dtype='int32')
        return int(factsSize), facts

    # Generally, get predicates after sampled.
    @staticmethod
    def get_pre(BENCHMARK, filename):
        with open(filename + BENCHMARK + "/relation2id.txt") as f:
            preSize = f.readline()
            pre = []
            predicateName = []
            for line in f.readlines():
                pre.append(line.strip('\n').split("	"))
                predicateName.append(line.split("	")[0])
        return predicateName, pre

    @staticmethod
    def get_fact_dic(pre_sample, facts_all, isUncertian):
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
                    if isUncertian is True:
                        temp_list1.append([facts_all[i, 0], facts_all[i, 1], facts_all[i, 3]])
                        temp_list2.append([facts_all[i, 1], facts_all[i, 0], facts_all[i, 3]])
                    else:
                        temp_list1.append([facts_all[i, 0], facts_all[i, 1]])
                        temp_list2.append([facts_all[i, 1], facts_all[i, 0]])
                    fact_dic[int(pre_sample[2 * j][0])] = temp_list1
                    fact_dic[int(pre_sample[2 * j + 1][0])] = temp_list2
        return fact_dic

    def search_and_evaluate(self, f, length, BENCHMARK, nowPredicate, ent_emb, rel_emb, dimension,
                            ent_size_all, fact_dic, DEGREE, isUncertain, _syn, _coocc):
        self.pt = nowPredicate[0]
        self.fact_dic_all = fact_dic
        self.entity_size_all = ent_size_all
        self.length = length
        self.isUncertian = isUncertain
        self._syn = _syn
        self._coocc = _coocc
        print("Length = %d." % self.length)
        relsize = rel_emb.shape[0]
        if f == 0:
            rel_emb = np.reshape(rel_emb, [relsize, dimension, dimension])
        # print(relation.shape)  # (-1, 100, 100) or (-1, 100)
        # print(entity.shape)  # (-1, 100)

        # Score Function
        candidate = []
        all_candidate_set = []  # Eliminate duplicate indexes.
        top_candidate_size = int(pow(relsize, length) * _syn)
        score_top_container = np.zeros(shape=(top_candidate_size, self.length+1))
        print("The number of SYN Top Candidates is %d" % top_candidate_size)

        # calculate the f1
        print("\nBegin to calculate the f1: synonymy")
        self.score_function1(f, score_top_container, rel_emb)
        # Method 1: Top ones until it reaches the 100th. OMIT!
        # Method 2: Use two matrices to catch rules.
        print(" Begin to use syn to filter: ")
        for item in score_top_container:
            index = [int(item[i]) for i in range(self.length)]
            # print(index)
            if f == 0:  # matrix
                result, degree = self.evaluate_and_filter(index, DEGREE)
                if result != 0 and index not in all_candidate_set:
                    all_candidate_set.append(index)
                    candidate.append([index, result, degree])
            elif f == 1:  # vector
                # It needs to evaluate for all arranges of index.
                for i in itertools.permutations(index, self.length):
                    # Deduplicate.
                    _index = list(np.array(i))
                    if _index in all_candidate_set:
                        continue
                    result, degree = self.evaluate_and_filter(_index, DEGREE)
                    if result != 0:
                        all_candidate_set.append(_index)
                        candidate.append([_index, result, degree])
        if not gc.isenabled():
            gc.enable()
        del rel_emb, score_top_container
        gc.collect()
        gc.disable()

        # calculate the f2
        top_candidate_size = int(pow(relsize, length) * _coocc)
        score_top_container = np.zeros(shape=(top_candidate_size, self.length+1))
        print("The number of COOCC Top Candidates is %d" % top_candidate_size)
        factsSize, facts = self.get_facts(BENCHMARK, filename="./sampled/")
        print("\nBegin to calculate the f2: Co-occurrence")
        self.score_function2(score_top_container, relsize, facts, ent_emb)

        print("\n Begin to use coocc to filter: ")
        for item in score_top_container:
            index = [int(item[i]) for i in range(self.length)]
            if index not in all_candidate_set:
                result, degree = self.evaluate_and_filter(index, DEGREE)
                if result != 0:
                    candidate.append([index, result, degree])
                    all_candidate_set.append(index)

        if not gc.isenabled():
            gc.enable()
        del ent_emb, score_top_container
        gc.collect()
        gc.disable()

        # Evaluation is still a cue method!
        print("\n*^_^* Yeah, there are %d rules. *^_^*\n" % len(candidate))

        # learn_weights(candidate)

        return candidate
