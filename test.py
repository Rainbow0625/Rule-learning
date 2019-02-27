import numpy as np
pre_sample = [[0, "a", 1], [2, "b", 3], [0, "c", 5]]

pre_sample_index = np.array([[pre[0], pre[2]] for pre in pre_sample], dtype=np.int32)
print(pre_sample_index)
a = pre_sample_index[:, 0]
print(a)
print(np.where(a==0)[0])


