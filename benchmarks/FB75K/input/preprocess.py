import numpy as np

if __name__=="__main__":
    # FACT
    facts = np.zeros(shape=(1, 3), dtype=np.int32)
    for file_type in ["test", "train"]:
        for i in range(1):  # 0-12
            filename = 'KG_' + file_type + '_d18_' + str(i) + '.txt'
            with open(filename, 'r') as f:
                facts_part = np.array([line.strip('\n').split('	') for line in f.readlines()], dtype=np.int32)
                # print(facts_part)
                facts = np.concatenate((facts, facts_part), axis=0)
    facts = np.delete(facts, 0, axis=0)
    print("fact size: %d" % len(facts))

    # ENTITY and PREDICATE
    entity = set()
    predicate = set()
    fact_list = []
    for fact in facts:
        entity.add(fact[0])
        entity.add(fact[2])
        predicate.add(fact[1])
        if [fact[0], fact[2], fact[1]] not in fact_list:
            fact_list.append([fact[0], fact[2], fact[1]])
    print("entity size: %d" % len(entity))
    print(entity)
    print("predicate size: %d" % len(predicate))
    print(predicate)
    print("FACT %d" % len(fact_list))
    print(fact_list)

    # Save in file.
    with open('Fact.txt', 'w') as f:
        f.write("%d\n" % len(fact_list))
        for fact in fact_list:
            f.write("%d %d %d\n" % (fact[0]-1, fact[1]-1, fact[2]-1))  # They begin from 1!

    with open('entity2id.txt', 'w') as f:
        f.write("%d\n" % len(entity))

    with open('relation2id.txt', 'w') as f:
        f.write("%d\n" % len(predicate))
        for i in range(len(predicate)):
            f.write('%s	%d\n' % ("pre_"+str(i), i))


