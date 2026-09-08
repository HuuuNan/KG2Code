import pickle

hop1=dict()

with open('../wikidata5m/wikidata5m_all_triplet.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        if hop1.get(line[0]) is None:
            hop1[line[0]]=set()
        if hop1.get(line[2]) is None:
            hop1[line[2]]=set()
        hop1[line[0]].add((line[0],line[1],line[2]))
        hop1[line[2]].add((line[0],line[1],line[2]))

with open("1hop.pkl", "wb") as f:
    pickle.dump(hop1, f)