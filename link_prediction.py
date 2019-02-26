import pickle

BENCHMARK = "FB15K237"
Pt_list = [0]
for Pt in Pt_list:
    with open('./rule' + BENCHMARK + '/rule_' + str(Pt) + '.pk', 'rb') as fp:
        candidate = pickle.load(fp)
