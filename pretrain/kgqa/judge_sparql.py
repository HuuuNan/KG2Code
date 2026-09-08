import json
import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

random.seed(42)

with open("endict.pkl", "rb") as file:
    endict = pickle.load(file)
with open("redict.pkl", "rb") as file:
    redict = pickle.load(file)
with open("typedict.pkl","rb") as file:
    typedict = pickle.load(file)

data=json.load(open('subgraph.json','r',encoding='utf-8'))

def find_same_type(e):
    passtype=['disambiguation','wikimedia','category','categorization']
    entype=endict[e]['categories_id']
    enlabel=endict[e]['label']
    #for t in entype:
    # only use the first category
    if len(entype)==0:
        return None
    t=entype[0]
    # skip some meaningless type
    if endict[e].get('categories') is None or len(endict[e]['categories'])==0 or endict[e]['categories'][0] is None:
        return None
    t_name=endict[e]['categories'][0]
    for p in passtype:
        if p.lower() in t_name.lower():
            return None
    if typedict.get(t) and len(typedict[t])>1:
        replace_e=enlabel
        while replace_e==enlabel:
            replace_e=random.choice(list(typedict[t]))
        return replace_e
    return None

G1='''SELECT DISTINCT ?x
WHERE {{
{e1} {r1} ?x .
}}'''

G2='''SELECT DISTINCT ?x
WHERE {{
?x {r1} {e1} .
}}'''

G3='''SELECT DISTINCT ?x
WHERE {{
{e1} {r1} ?x .
?x {r2} {e2} .
}}'''

G4='''SELECT DISTINCT ?x
WHERE {{
?x {r1} {e1} .
?x {r2} {e2} .
}}'''

G5='''SELECT DISTINCT ?x
WHERE {{
?x {r1} {e1} .
?x {r2} {e2} .
?x {r3} {e3} .
}}'''

G6='''SELECT DISTINCT ?x
WHERE {{
{e1} {r1} e2 .
e2 {r2} ?x .
}}'''

G7='''SELECT DISTINCT ?x
WHERE {{
{e1} {r1} e2 .
e2 {r2} e3 .
e3 {r3} ?x .
}}'''

G8='''SELECT DISTINCT ?x
WHERE {{
{e1} {r1} e2 .
e2 {r2} e3 .
e3 {r3} e4 .
e4 {r4} ?x .
}}'''

A1='''ASK
{{
{e1} {r1} {a} .
}}'''

A2='''ASK
{{
{a} {r1} {e1} .
}}'''

A3='''ASK
{{
{e1} {r1} {a} .
{a} {r2} {e2} .
}}'''

A4='''ASK
{{
{a} {r1} {e1} .
{a} {r2} {e2} .
}}'''

A5='''ASK
{{
{a} {r1} {e1} .
{a} {r2} {e2} .
{a} {r3} {e3} .
}}'''

A6='''ASK
{{
{e1} {r1} e2 .
e2 {r2} {a} .
}}'''

C1='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
{e1} {r1} ?x .
}}'''

C2='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
?x {r1} {e1} .
}}'''

C3='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
{e1} {r1} ?x .
?x {r2} {e2} .
}}'''

C4='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
?x {r1} {e1} .
?x {r2} {e2} .
}}'''

C5='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
?x {r1} {e1} .
?x {r2} {e2} .
?x {r3} {e3} .
}}'''

C6='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
{e1} {r1} e2 .
e2 {r2} ?x .
}}'''

C7='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
{e1} {r1} e2 .
e2 {r2} e3 .
e3 {r3} ?x .
}}'''

C8='''SELECT (COUNT(DISTINCT ?x) AS ?count)
WHERE {{
{e1} {r1} e2 .
e2 {r2} e3 .
e3 {r3} e4 .
e4 {r4} ?x .
}}'''

def process_sample(sample):
    valid=True
    # name for head entity, answer, subgraph, relation path 
    # name for head entity
    head_name=[]
    for e in sample['head entity']:
        if endict[e]['label'] is not None:
            head_name.append(endict[e]['label'])
        else:
            valid=False
            break
    if not valid:
        return None
    # name for answer
    answer_name=[]
    for a in sample['answer']:
        if endict[a]['label'] is not None:
            answer_name.append(endict[a]['label'])
        else:
            valid=False
            break
    if not valid:
        return None
    # name for subgraph
    subgraph_name=[]
    for t in sample['subgraph']:
        t_name=[]
        if endict[t[0]]['label'] is not None:
            t_name.append(endict[t[0]]['label'])
        else:
            t_name.append(t[0])
        if redict[t[1]]['label'] is not None:
            t_name.append(redict[t[1]]['label'])
        else:
            t_name.append(t[1])
        if endict[t[2]]['label'] is not None:
            t_name.append(endict[t[2]]['label'])
        else:
            t_name.append(t[2])
        subgraph_name.append(t_name)
    # name for relation path
    path_name=[]
    for r in sample['relation path']:
        if redict[r]['label'] is not None:
            path_name.append(redict[r]['label'])
        else:
            valid=False
            break
    if not valid:
        return None
    ans=random.choice(answer_name)
    ans_mid=sample['answer'][answer_name.index(ans)]
    judge_type=random.choice(['yes','no'])      
    sample['head entity name']=head_name
    sample['answer']=[judge_type]
    sample['answer name']=[judge_type]
    sample['subgraph name']=subgraph_name
    sample['relation path name']=path_name
    if judge_type=='yes':       
        if sample["graph id"]==1:
            sample['sparql']=A1.format(e1=head_name[0],r1=path_name[0],a=ans)
        if sample["graph id"]==2:
            sample['sparql']=A2.format(e1=head_name[0],r1=path_name[0],a=ans)
        if sample["graph id"]==3:
            sample['sparql']=A3.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1],a=ans)
        if sample["graph id"]==4:
            sample['sparql']=A4.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1],a=ans)
        if sample["graph id"]==5:
            sample['sparql']=A5.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1],e3=head_name[2],r3=path_name[2],a=ans)
        if sample["graph id"]==6:
            sample['sparql']=A6.format(e1=head_name[0],r1=path_name[0],r2=path_name[1],a=ans)
        if sample["graph id"]==7:
            return None
        if sample["graph id"]==8:
            return None
    else:
        if sample["graph id"]==1:
            e1=head_name[0]
            e1_mid=sample["head entity"][0]
            # replace e1 or ans. 1 for e1, 2 for ans.
            rid=random.randint(1,2)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None
            sample['sparql']=A1.format(e1=e1,r1=path_name[0],a=ans)
        if sample["graph id"]==2:
            e1=head_name[0]
            e1_mid=sample["head entity"][0]
            # replace e1 or ans. 1 for e1, 2 for ans.
            rid=random.randint(1,2)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None        
            sample['sparql']=A2.format(e1=e1,r1=path_name[0],a=ans)  
        if sample["graph id"]==3:
            e1=head_name[0]
            e1_mid=sample["head entity"][0]
            e2=head_name[1]
            e2_mid=sample["head entity"][1]
            # replace e1, e2 or ans. 1 for e1, 2 for e2, 3 for ans.
            rid=random.randint(1,3)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                e2=find_same_type(e2_mid)
                if e2 is None:
                    return None
            if rid==3:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None
            sample['sparql']=A3.format(e1=e1,r1=path_name[0],e2=e2,r2=path_name[1],a=ans)                  
        if sample["graph id"]==4:
            e1=head_name[0]
            e2=head_name[1]
            e1_mid=sample["head entity"][0]
            e2_mid=sample["head entity"][1]
            # replace e1, e2 or ans. 1 for e1, 2 for e2, 3 for ans.
            rid=random.randint(1,3)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                e2=find_same_type(e2_mid)
                if e2 is None:
                    return None
            if rid==3:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None
            sample['sparql']=A4.format(e1=e1,r1=path_name[0],e2=e2,r2=path_name[1],a=ans)
        if sample["graph id"]==5:
            e1=head_name[0]
            e1_mid=sample["head entity"][0]
            e2=head_name[1]
            e2_mid=sample["head entity"][1]
            e3=head_name[2]
            e3_mid=sample["head entity"][2]
            # replace e1, e2 or ans. 1 for e1, 2 for e2, 3 for e3, 4 for ans.
            rid=random.randint(1,4)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                e2=find_same_type(e2_mid)
                if e2 is None:
                    return None
            if rid==3:
                e3=find_same_type(e3_mid)
                if e3 is None:
                    return None
            if rid==4:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None
            sample['sparql']=A5.format(e1=e1,r1=path_name[0],e2=e2,r2=path_name[1],e3=e3,r3=path_name[2],a=ans)
        if sample["graph id"]==6:
            e1=head_name[0]
            e1_mid=sample["head entity"][0]
            # replace e1 or ans. 1 for e1, 2 for ans.
            rid=random.randint(1,2)
            if rid==1:
                e1=find_same_type(e1_mid)
                if e1 is None:
                    return None
            if rid==2:
                ans=find_same_type(ans_mid)
                if ans is None:
                    return None
            sample['sparql']=A6.format(e1=e1,r1=path_name[0],r2=path_name[1],a=ans)  
        if sample["graph id"]==7:
            return None
        if sample["graph id"]==8:
            return None
    return sample

if __name__ == "__main__":
    with Pool(processes=30) as pool:
        results = list(tqdm(pool.imap_unordered(process_sample, data), total=len(data)))
        
    corpus = [r for r in results if r is not None]

    with open('judge_sparql.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)                   