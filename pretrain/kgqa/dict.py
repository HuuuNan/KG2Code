import pickle

out_rel=dict()
in_rel=dict()
out_tri=dict()
in_tri=dict()

with open('../wikidata5m/wikidata5m_all_triplet.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        if out_rel.get(line[0]) is None:
            out_rel[line[0]]=set()
        out_rel[line[0]].add(line[1])
        if in_rel.get(line[2]) is None:
            in_rel[line[2]]=set()
        in_rel[line[2]].add(line[1])
        if out_tri.get((line[0],line[1])) is None:
            out_tri[(line[0],line[1])]=set()
        out_tri[(line[0],line[1])].add(line[2])
        if in_tri.get((line[2],line[1])) is None:
            in_tri[(line[2],line[1])]=set()
        in_tri[(line[2],line[1])].add(line[0])

with open("out_relation.pkl", "wb") as f:
    pickle.dump(out_rel, f)
    
with open("in_relation.pkl", "wb") as f:
    pickle.dump(in_rel, f)
    
with open("out_triple.pkl", "wb") as f:
    pickle.dump(out_tri, f)

with open("in_triple.pkl", "wb") as f:
    pickle.dump(in_tri, f)