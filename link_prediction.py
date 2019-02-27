import pickle
import time
# Run it alone.
BENCHMARK = "FB15K237"

if __name__ == '__main__':
    begin = time.time()
    Pt_list = [0]
    for Pt in Pt_list:
        with open('./rule' + BENCHMARK + '/rule_' + str(Pt) + '.pk', 'rb') as fp:
            candidate = pickle.load(fp)  # [index, flag={1:Rule, 2:Quality Rule}, degree]
        # get fact_dict 稀疏矩阵的乘法



    end = time.time()
    print("Total time: %f" % begin-end)