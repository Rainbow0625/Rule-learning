import time
import numpy as np
from scipy import sparse
import model_learn_weights as mlw
import itertools
import gc
import sys


class RSALW(object):
    def __int__(self):
        self.pt = None
        self.fact_dic_sample = None
        self.fact_dic_all = None
        self.ent_size_sample = None
        self.ent_size_all = None
        self.length = None
        self.isUncertian = False
        self._syn = None
        self._coocc = None
        self.P_i = None
        self.ptmatrix_part = None
        self.ptmatrix_full = None
        self.ptmatrix_sample = None

    @staticmethod
    def sim(para1, para2):  # similarity of vector or matrix
        return np.e ** (-np.linalg.norm(para1 - para2, ord=2))

    @staticmethod
    def get_fact_dic_sample(facts_sample, isUncertian=False):
        # Only save once for the reverse pre.e.g. 0, 2, 4....
        fact_dic = {}
        for f in facts_sample:
            if f[2] % 2 == 0:
                if f[2] in fact_dic.keys():
                    templist = fact_dic.get(f[2])
                else:
                    templist = []
                templist.append([f[0], f[1]])
                fact_dic[f[2]] = templist
        return fact_dic

    @staticmethod
    def get_fact_dic_all(pre_sample, facts_all):
        # Only save once for the reverse pre.e.g. 0, 2, 4....
        # fact_dic: key: P_index_new , value: all_fact_list
        # pre_sample_index = np.array([[pre[0], pre[2]] for pre in pre_sample], dtype=np.int32)
        # old_index_p = pre_sample_index[:, 1]
        old_index_p = np.array([pre[2] for pre in pre_sample], dtype=np.int32)
        fact_dic = {}
        for f in facts_all:
            if f[2] in set(old_index_p):
                new_index = np.where(old_index_p == f[2])[0][0]  # It must be even.
                # new_index = pre_sample_index[np.where(old_index_p == f[2])[0][0]][0]
                if new_index in fact_dic.keys():
                    temp_list = fact_dic.get(new_index)
                else:
                    temp_list = []
                temp_list.append([f[0], f[1]])
                fact_dic[new_index] = temp_list
        return fact_dic

    def is_repeated(self, M_index):
        for i in range(1, self.length):
            if M_index[i] < M_index[0]:
                return True
        return False

    def get_index_tuple(self):
        max_i = int((self.length + 1) / 2)
        a = [x for x in range(1, max_i + 1)]
        if self.length % 2 == 0:  # even
            b = a.copy()
        else:  # odd
            b = a.copy()
            b.pop()
        b.reverse()
        a.extend(b)
        P_cartprod_list = [self.P_i[i] for i in a]
        self.index_tuple_size = 1
        for item in P_cartprod_list:
            self.index_tuple_size = self.index_tuple_size * len(item)
        print("\nindex_tuple_size: %d" % self.index_tuple_size)
        self.index_tuple = itertools.product(*P_cartprod_list)

    def get_subandobj_dic_for_f2(self):
        # For predicates: 0, 2, 4, ... subdic, objdic
        # For predicates: 1, 3, 5, ... objdic, subdic
        objdic = {}  # key:predicate value: set
        subdic = {}  # key:predicate value: set
        # print(len(self.fact_dic_sample))
        for key in self.fact_dic_sample.keys():
            tempsub = set()
            tempobj = set()
            facts_list = self.fact_dic_sample.get(key)
            for f in facts_list:
                tempsub.add(f[0])
                tempobj.add(f[1])
            subdic[key] = tempsub
            objdic[key] = tempobj
        return subdic, objdic

    def score_function1(self, flag, score_top_container, relation):  # synonymy!
        for index in self.index_tuple:
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

    def score_function2(self, score_top_container, entity, sub_dic, obj_dic):  # co-occurrence
        tt = time.time()
        # get the average vector of average predicate which is saved in the dictionary.
        average_vector = {}
        for key in sub_dic:
            # print(key)
            sub = sum([entity[item, :] for item in sub_dic[key]]) / len(sub_dic[key])
            obj = sum([entity[item, :] for item in obj_dic[key]]) / len(obj_dic[key])
            # For predicates: 0, 2, 4, ... [sub, obj]
            # For predicates: 1, 3, 5, ... [obj, sub]
            average_vector[key] = [sub, obj]
            average_vector[key + 1] = [obj, sub]
        # print("\n the dic's size is equal to the predicates' number! ")
        # print(len(average_vector))
        f = 0
        for index in self.index_tuple:
            sys.stdout.write('\rProgress: %d - %d ' % (f, self.index_tuple_size))
            sys.stdout.flush()
            f = f + 1
            para_sum = float(0)
            for i in range(self.length - 1):
                para_sum = para_sum + self.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
            value = para_sum + self.sim(average_vector.get(index[0])[0], average_vector.get(self.pt)[0]) \
                    + self.sim(average_vector.get(index[self.length - 1])[1],
                               average_vector.get(self.pt)[1])
            top_values = score_top_container[:, self.length]
            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                for i in range(self.length):
                    score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
        print("Time: %f." % (time.time()-tt))

    def getmatrix(self, p, isWhichKG):
        # sparse matrix
        re_flag = False
        if p % 2 == 1:
            p = p - 1
            re_flag = True
        # Pt: avoid cal it again.
        if p == self.pt:
            if isWhichKG == 0:
                if self.ptmatrix_sample != None:
                    return self.ptmatrix_sample
            elif isWhichKG == 1:
                if self.ptmatrix_part != None:
                    return self.ptmatrix_part
            elif isWhichKG == 2:
                if self.ptmatrix_full != None:
                    return self.ptmatrix_full
        if isWhichKG == 0:
            pfacts = self.fact_dic_sample.get(p)
            pmatrix = sparse.dok_matrix((self.ent_size_sample, self.ent_size_sample), dtype=np.int8)
            if re_flag:
                for f in pfacts:
                    pmatrix[f[1], f[0]] = 1
            else:
                for f in pfacts:
                    pmatrix[f[0], f[1]] = 1
        elif isWhichKG == 1:  # Evaluate on Pt's entity one-hot matrix.
            pfacts = self.fact_dic_all.get(p)
            ent_size = len(self.E_0)
            E_0_list = list(self.E_0)
            pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int8)
            for f in pfacts:
                if f[0] in self.E_0 and f[1] in self.E_0:
                    if re_flag:
                        pmatrix[E_0_list.index(f[1]), E_0_list.index(f[0])] = 1
                    else:
                        pmatrix[E_0_list.index(f[0]), E_0_list.index(f[1])] = 1
        else:
            pfacts = self.fact_dic_all.get(p)
            pmatrix = sparse.dok_matrix((self.ent_size_all, self.ent_size_all), dtype=np.int8)
            if re_flag:
                for f in pfacts:
                    pmatrix[f[1], f[0]] = 1
            else:
                for f in pfacts:
                    pmatrix[f[0], f[1]] = 1
        return pmatrix

    def calSupportGreaterThanOne(self, pmatrix, isSampleKG):
        if isSampleKG:
            ptmatrix = self.ptmatrix_sample
        else:
            ptmatrix = self.ptmatrix_part
        supp = 0
        head = len(ptmatrix)
        body = len(pmatrix)
        if head == 0 or body == 0:
            return False
        if head < body:
            for key in ptmatrix.keys():
                if pmatrix.get(key) > 0:
                    supp += 1
                    return True
        if head >= body:
            for key in pmatrix.keys():
                if ptmatrix.get(key) == 1:
                    supp += 1
                    return True
        return False

    def calSCandHC_csr(self, pmatrix):
        print("\n---------------csr---------------\n")
        # calculate New SC
        # supp_score = 0.0
        # body_score = 0.0
        ptmatrix = self.ptmatrix_full
        head = len(ptmatrix)
        body = pmatrix.nnz
        supp = 0
        if head == 0 or body == 0:
            return 0, 0
        row_compress = pmatrix.indptr
        col = pmatrix.indices
        # print(pmatrix.nnz)
        # print(sys.getsizeof(pmatrix))
        flag = 0
        for i in range(pmatrix.shape[0]):
            row_num = row_compress[i + 1] - row_compress[i]
            if row_num == 0:
                continue
            row_col = col[flag: flag + row_num]
            for j in range(row_num):
                if ptmatrix.get(tuple([i, row_col[j]])) == 1:
                    supp = supp + 1
            flag += row_num
        # Judge by supp.
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        return SC, HC

    def calSCandHC_dok(self, pmatrix):
        ptmatrix = self.ptmatrix_full
        head = len(ptmatrix)
        body = len(pmatrix)
        supp = 0
        # calculate New SC
        # supp_score = 0.0
        # body_score = 0.0
        if head == 0 or body == 0:
            return 0, 0
        if head < body:
            for key in ptmatrix.keys():
                if pmatrix.get(key) > 0:
                    supp = supp + 1
        elif head >= body:
            for key in pmatrix.keys():
                if ptmatrix.get(key) == 1:
                    supp = supp + 1
        # Judge by supp.
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        return SC, HC

    def matrix_dot(self, index, isWhichKG):  # 0:sample 1:part 2:fu;
        pmatrix = self.getmatrix(index[0], isWhichKG)
        for i in range(1, self.length):
            # Matrix distribution law
            pmatrix = pmatrix.dot(self.getmatrix(index[i], isWhichKG))
            if not gc.isenabled():
                gc.enable()
            gc.collect()
            gc.disable()
        return pmatrix

    def evaluate_and_filter(self, index, DEGREE, isfullKG):
        if not isfullKG:
            if len(self.E_0) <= 100000:
                # On part. (Priority)
                pmatrix = self.matrix_dot(index, 1)
                pmatrix = pmatrix.todok()
                if not self.calSupportGreaterThanOne(pmatrix, False):
                    return 0, None
            else:
                # On sample.
                pmatrix = self.matrix_dot(index, 0)
                pmatrix = pmatrix.todok()
                if not self.calSupportGreaterThanOne(pmatrix, True):
                    return 0, None
        # On full.
        pmatrix = self.matrix_dot(index, 2)
        # calculate the temp SC and HC
        if sys.getsizeof(pmatrix) > 10485760:  # 10M
            # Type of pmatrix:  csr_matrix!
            print(sys.getsizeof(pmatrix))
            print("Date size:")
            print("pmatrix len:%d" % pmatrix.nnz)
            if isfullKG:
                print("full:")
                print(pmatrix.nnz / self.ent_size_all ** 2)
            else:
                print(pmatrix.nnz / self.ent_size_sample ** 2)
            print("\n")
            SC, HC = self.calSCandHC_csr(pmatrix)
        else:
            pmatrix = pmatrix.todok()
            # Type of pmatrix:  dok_matrix!
            # print("Date size:")
            # print("pmatrix len:%d" % len(pmatrix))
            # if isfullKG:
            # print("full:")
            # print(len(pmatrix) / self.ent_size_all ** 2)
            # else:
            # print(len(pmatrix) / self.ent_size_sample ** 2)
            # print("\n")
            SC, HC = self.calSCandHC_dok(pmatrix)
        degree = [SC, HC]
        # print(degree)
        if SC >= DEGREE[0] and HC >= DEGREE[1]:
            # 1: quality rule
            # 2: high quality rule
            print("\n%s - HC:%s, SC:%s." % (str(index), str(HC), str(SC)))
            # print("The NEW Standard Confidence of this rule is " + str(NSC))
            if SC >= DEGREE[2] and HC >= DEGREE[3]:
                print("WOW, a quality rule!")
                return 2, degree
            return 1, degree
        return 0, None

    # def learn_weights(self, candidate):
    #     # In the whole data set to learn the weights.
    #     training_Iteration = 100
    #     learning_Rate = 0.1
    #     regularization_rate = 0.1
    #     model = mlw.LearnModel()
    #     # fact_dic_sample!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     model.__int__(self.length, training_Iteration, learning_Rate, regularization_rate,
    #                   self.fact_dic_sample, self.ent_size_sample, candidate, self.pt, self.isUncertian)
    #
    #     model.train()

    def search_and_evaluate(self, isUncertain, f, length, dimension, DEGREE, nowPredicate,
                            ent_emb, rel_emb, _syn, _coocc, P_new_index_list, isfullKG,
                            fact_dic_sample, fact_dic_all, ent_size_sample, ent_size_all, E_0):
        self.pt = nowPredicate[0]
        self.fact_dic_sample = fact_dic_sample
        self.fact_dic_all = fact_dic_all
        self.ent_size_sample = ent_size_sample
        self.ent_size_all = ent_size_all
        self.length = length
        self.isUncertian = isUncertain
        self._syn = _syn
        self._coocc = _coocc
        self.P_i = P_new_index_list
        self.E_0 = E_0
        print("Length = %d." % self.length)
        relsize = rel_emb.shape[0]
        if f == 0:
            rel_emb = np.reshape(rel_emb, [relsize, dimension, dimension])
        # print(relation.shape)  # (-1, 100, 100) or (-1, 100)
        # print(entity.shape)  # (-1, 100)

        # Score Function
        candidate = []
        all_candidate_set = []  # Eliminate duplicate indexes.

        if not gc.isenabled():
            gc.enable()
        del rel_emb
        gc.collect()
        gc.disable()

        # Get index tuple.
        self.get_index_tuple()

        # Calculate the f2.
        # top_candidate_size = int(_coocc * self.index_tuple_size)
        if self.index_tuple_size < _coocc:
            top_candidate_size = self.index_tuple_size
        else:
            top_candidate_size = _coocc
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        print("The number of COOCC Top Candidates is %d" % top_candidate_size)
        subdic, objdic = self.get_subandobj_dic_for_f2()
        print("\nBegin to calculate the f2: Co-occurrence")
        self.score_function2(score_top_container, ent_emb, subdic, objdic)

        if not gc.isenabled():
            gc.enable()
        del ent_emb, subdic, objdic
        gc.collect()
        gc.disable()

        print("\nPt matrix:")
        if not isfullKG:
            if len(self.E_0) > 100000:
                self.ptmatrix_sample = self.getmatrix(self.pt, 0)
                print(" Sample: len:%d  size:%d" % (len(self.ptmatrix_sample), sys.getsizeof(self.ptmatrix_sample)))
            else:
                self.ptmatrix_part = self.getmatrix(self.pt, 1)
                print(" Part: len:%d  size:%d" % (len(self.ptmatrix_part), sys.getsizeof(self.ptmatrix_part)))
        self.ptmatrix_full = self.getmatrix(self.pt, 2)
        print(" Full: len:%d  size:%d" % (len(self.ptmatrix_full), sys.getsizeof(self.ptmatrix_full)))

        print("\nBegin to use coocc to filter: ")
        count = 0
        tt = time.time()
        for item in score_top_container:
            count += 1
            sys.stdout.write('\rProgress: %d - %d ' % (count, top_candidate_size))
            sys.stdout.flush()
            index = [int(item[i]) for i in range(self.length)]
            if index not in all_candidate_set:
                result, degree = self.evaluate_and_filter(index, DEGREE, isfullKG)
                if result != 0:
                    candidate.append([index, result, degree])
                    all_candidate_set.append(index)
        print("Time:%f." % (time.time()-tt))
        if not gc.isenabled():
            gc.enable()
        del score_top_container
        gc.collect()
        gc.disable()

        '''
        # Calculate the f1.
        # top_candidate_size = int(_syn * self.index_tuple_size)
        if self.index_tuple_size < _syn:
            top_candidate_size = self.index_tuple_size
        else:
            top_candidate_size = _syn
        top_candidate_size = _syn
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        print("The number of SYN Top Candidates is %d" % top_candidate_size)
        print("\nBegin to calculate the f1: synonymy")
        self.score_function1(f, score_top_container, rel_emb)
        # Method 1: Top ones until it reaches the 100th. OMIT!
        # Method 2: Use two matrices to catch rules.
        print("\n Begin to use syn to filter: ")
        for item in score_top_container:
            index = [int(item[i]) for i in range(self.length)]
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
                    result, degree = self.evaluate_and_filter(_index, DEGREE, isfullKG)
                    if result != 0:
                        all_candidate_set.append(_index)
                        candidate.append([_index, result, degree])
        if not gc.isenabled():
            gc.enable()
        del rel_emb, score_top_container
        gc.collect()
        gc.disable()
        '''

        print("\n*^_^* Yeah, there are %d rules. *^_^*." % len(candidate))

        # learn_weights(candidate)
        return candidate
