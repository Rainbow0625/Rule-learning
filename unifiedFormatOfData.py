from typing import Dict, Any

import scipy.io as sio


def format_predicate(path):
    with open(path + '/predindex.txt', 'r') as f:
        predicateName = [line.strip('\n').strip('[').strip(']').split(', ')[1].strip('\'') for line in
                         f.readlines()]
        size = len(predicateName)
        print("Total predicates:%d" % size)
        f.close()
    with open(path + '/relation2id.txt', 'w') as f:
        f.write(str(size)+"\n")
        for i in range(size):
            f.write(str(predicateName[i])+"	"+str(i)+"\n")
        f.close()
    return True


def format_fact_entity(path):
    data = sio.loadmat(path + '/tensord.mat')
    # print(data.keys())
    # dict_keys(['__header__', '__version__', '__globals__', 'vals', 'rattr', 'eattr', 'subs', 'size'])
    size = data.get('size')
    # eg: DBpedia: [head entity size:3102999, tail entity size:3102999, predicate size:650]
    print("Data size:" + str(size))
    entity_size = int(str(size[0]).strip('[').strip(']'))
    print("Total entity:%d" % entity_size)
    fact = data.get('subs')
    ent_set = set()
    fact_size = len(fact)
    print("Total facts:%d" % fact_size)
    with open(path+'/entity2id.txt', 'w') as f:
        f.write(str(entity_size))
        f.close()
    with open(path + '/Fact.txt', 'w') as f:
        f.write(str(fact_size)+'\n')
        for i in range(fact_size):
            ent_set.add(int(fact[i, 0]-1))
            ent_set.add(int(fact[i, 1]-1))
            line = str(fact[i, 0]-1) + ' ' + str(fact[i, 1]-1) + ' ' + str(fact[i, 2]-1)+'\n'
            f.write(line)
        f.close()
    print("entity size: %d" % len(ent_set))
    return True


if __name__ == "__main__":
    # "DB"  "Wiki"  "Yago"
    BENCHMARK = "DB"
    path = './benchmarks/' + BENCHMARK
    if format_predicate(path):
        print("Format predicate ok!")
    if format_fact_entity(path):
        print("Format entity ok!")


