import pickle
import random
import json
from tqdm import tqdm

random.seed(42)

with open("out_relation.pkl", "rb") as file:
    out_rel = pickle.load(file)
    
with open("in_relation.pkl", "rb") as file:
    in_rel = pickle.load(file)
    
with open("out_triple.pkl", "rb") as file:
    out_tri = pickle.load(file)
    
with open("in_triple.pkl", "rb") as file:
    in_tri = pickle.load(file)
    
# given an entity list, return the 1-hop relations of these entity
def get_relation(enlist,direction):
    if direction=='forward':
        reset=set()
        for e in enlist:
            if out_rel.get(e):
                reset=reset.union(out_rel[e])
    if direction=='backward':
        reset=set()
        for e in enlist:
            if in_rel.get(e):
                reset=reset.union(in_rel[e])
    return list(reset)
  
# given an entity list and a relation, return the tail entity list
def get_entity(enlist,rel,direction):
    if direction=='forward':
        enset=set()
        for e in enlist:
            if out_tri.get((e,rel)):
                enset=enset.union(out_tri[(e,rel)])
    if direction=='backward':
        enset=set()
        for e in enlist:
            if in_tri.get((e,rel)):
                enset=enset.union(in_tri[(e,rel)])
    return list(enset)
 
# collect triple path for relation_chain
def find_all_paths(start_entities, relation_chain, kg_dict):
    all_paths = []

    def dfs(current_entity, relation_index, path_so_far):
        if relation_index == len(relation_chain):
            all_paths.append(path_so_far)
            return

        relation = relation_chain[relation_index]
        next_entities = kg_dict.get((current_entity, relation), [])
        
        for tail in next_entities:
            triple = (current_entity, relation, tail)
            dfs(tail, relation_index + 1, path_so_far + [triple])

    for entity in start_entities:
        dfs(entity, 0, [])

    return all_paths

enlist=[]  
with open('entity.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        enlist.append(line.strip())

def construct(e,gid):
    sample=dict()
    subgraph=set()
    if gid==1:
        relist=get_relation([e],'forward')
        if len(relist)==0:
            return sample
        rel=random.choice(relist)
        anslist=get_entity([e],rel,'forward')
        if len(anslist)==0:
            return sample
        sample["graph id"]=1
        sample["head entity"]=[e]
        sample["answer"]=anslist
        for a in anslist:
            subgraph.add((e,rel,a))
        sample["subgraph"]=list(subgraph)
        sample["relation path"]=[rel]
    if gid==2:
        relist=get_relation([e],'backward')
        if len(relist)==0:
            return sample
        rel=random.choice(relist)
        anslist=get_entity([e],rel,'backward')
        if len(anslist)==0:
            return sample
        sample["graph id"]=2
        sample["head entity"]=[e]
        sample["answer"]=anslist
        for a in anslist:
            subgraph.add((a,rel,e))
        sample["subgraph"]=list(subgraph) 
        sample["relation path"]=[rel]
    if gid==3:
        relist1=get_relation([e],'forward')
        if len(relist1)==0:
            return sample  
        rel1=random.choice(relist1)
        mid1=get_entity([e],rel1,'forward')
        relist2=get_relation(mid1,'forward')
        if len(relist2)==0:
            return sample
        rel2=random.choice(relist2)
        mid2=get_entity(mid1,rel2,'forward')
        head1=e
        # random choice one entity from mid2 for head entity 2
        head2=random.choice(mid2)
        anslist=list(set(mid1).intersection(set(get_entity([head2],rel2,'backward'))))
        if len(anslist)==0:
            return sample
        sample["graph id"]=3        
        sample["head entity"]=[head1,head2]
        sample["answer"]=anslist
        for a in anslist:
            subgraph.add((head1,rel1,a))
            subgraph.add((a,rel2,head2))
        sample["subgraph"]=list(subgraph)
        sample["relation path"]=[rel1,rel2]
    if gid==4:
        relist1=get_relation([e],'backward')
        if len(relist1)==0:
            return sample
        rel1=random.choice(relist1)
        mid1=get_entity([e],rel1,'backward')
        relist2=get_relation(mid1,'forward')
        available_rels = list(set(relist2) - {rel1})
        if not available_rels:
            return sample
        rel2 = random.choice(available_rels)
        while rel2==rel1:
            rel2=random.choice(relist2)
        mid2=get_entity(mid1,rel2,'forward')
        head1=e
        # random choice one entity from mid2 for head entity 2
        head2=random.choice(mid2) 
        anslist=list(set(mid1).intersection(set(get_entity([head2],rel2,'backward'))))  
        if len(anslist)==0:
            return sample
        sample["graph id"]=4       
        sample["head entity"]=[head1,head2]
        sample["answer"]=anslist
        for a in anslist:
            subgraph.add((a,rel1,head1))
            subgraph.add((a,rel2,head2))
        sample["subgraph"]=list(subgraph)
        sample["relation path"]=[rel1,rel2]
    if gid==5:
        relist1=get_relation([e],'backward')
        if len(relist1)==0:
            return sample
        rel1=random.choice(relist1)
        mid1=get_entity([e],rel1,'backward')
        relist2=get_relation(mid1,'forward')
        available_rels = list(set(relist2) - {rel1})
        if len(available_rels) < 2:
            return sample
        rel2, rel3 = random.sample(available_rels, 2)
        mid2=get_entity(mid1,rel2,'forward')
        mid3=get_entity(mid1,rel3,'forward')
        head1=e
        # random choice one entity from mid2 for head entity 2, one entity from mid3 for head entity 3
        head2=random.choice(mid2)
        head3=random.choice(mid3) 
        anslist=list(set(mid1).intersection(set(get_entity([head2],rel2,'backward'))).intersection(set(get_entity([head3],rel3,'backward'))))
        if len(anslist)==0:
            return sample
        sample["graph id"]=5       
        sample["head entity"]=[head1,head2,head3]
        sample["answer"]=anslist
        for a in anslist:
            subgraph.add((a,rel1,head1))
            subgraph.add((a,rel2,head2))
            subgraph.add((a,rel3,head3))
        sample["subgraph"]=list(subgraph)
        sample["relation path"]=[rel1,rel2,rel3]
    if gid==6:
        relist1=get_relation([e],'forward')
        if len(relist1)==0:
            return sample
        rel1=random.choice(relist1)
        mid1=get_entity([e],rel1,'forward')
        relist2=get_relation(mid1,'forward')
        if len(relist2)==0:
            return sample
        rel2=random.choice(relist2)
        anslist=get_entity(mid1,rel2,'forward')
        if len(anslist)==0:
            return sample
        sample["graph id"]=6
        sample["head entity"]=[e]
        sample["answer"]=anslist
        sample["relation path"]=[rel1,rel2]
        paths=find_all_paths([e], [rel1,rel2], out_tri)
        unique_triples = set()
        for path in paths:
            for triple in path:
                unique_triples.add(triple)
        triple_list = list(unique_triples)
        sample["subgraph"]=triple_list
    if gid==7:
        relist1=get_relation([e],'forward')
        if len(relist1)==0:
            return sample
        rel1=random.choice(relist1) 
        mid1=get_entity([e],rel1,'forward')
        relist2=get_relation(mid1,'forward')
        if len(relist2)==0:
            return sample
        rel2=random.choice(relist2)
        mid2=get_entity(mid1,rel2,'forward')
        relist3=get_relation(mid2,'forward')
        if len(relist3)==0:
            return sample
        rel3=random.choice(relist3)
        anslist=get_entity(mid2,rel3,'forward')
        if len(anslist)==0:
            return sample
        sample["graph id"]=7
        sample["head entity"]=[e]
        sample["answer"]=anslist
        sample["relation path"]=[rel1,rel2,rel3]
        paths=find_all_paths([e], [rel1,rel2,rel3], out_tri)
        unique_triples = set()
        for path in paths:
            for triple in path:
                unique_triples.add(triple)
        triple_list = list(unique_triples)
        sample["subgraph"]=triple_list
    if gid==8:
        relist1=get_relation([e],'forward')
        if len(relist1)==0:
            return sample
        rel1=random.choice(relist1) 
        mid1=get_entity([e],rel1,'forward')
        relist2=get_relation(mid1,'forward')
        if len(relist2)==0:
            return sample
        rel2=random.choice(relist2)
        mid2=get_entity(mid1,rel2,'forward')
        relist3=get_relation(mid2,'forward')
        if len(relist3)==0:
            return sample
        rel3=random.choice(relist3)
        mid3=get_entity(mid2,rel3,'forward')
        relist4=get_relation(mid3,'forward')
        if len(relist4)==0:
            return sample
        rel4=random.choice(relist4)
        anslist=get_entity(mid3,rel4,'forward')
        if len(anslist)==0:
            return sample
        sample["graph id"]=8
        sample["head entity"]=[e]
        sample["answer"]=anslist
        sample["relation path"]=[rel1,rel2,rel3,rel4]
        paths=find_all_paths([e], [rel1,rel2,rel3,rel4], out_tri)
        unique_triples = set()
        for path in paths:
            for triple in path:
                unique_triples.add(triple)
        triple_list = list(unique_triples)
        sample["subgraph"]=triple_list 
    return sample

gid=1
ques_sub=[]
for e in tqdm(enlist):
    sample=construct(e,gid)
    gid1=gid
    while len(sample)==0 and gid1-gid<7:
        sample=construct(e,gid1 % 8+1)
        gid1 = gid1 + 1
    if len(sample)!=0:
        ques_sub.append(sample)
    gid = (gid % 8) + 1
    
print(len(ques_sub))
json.dump(ques_sub,open('subgraph.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)            