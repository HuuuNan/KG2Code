import json
import random
import pickle
from tqdm import tqdm

random.seed(42)
with open("endict.pkl", "rb") as file:
    endict = pickle.load(file)
with open("redict.pkl", "rb") as file:
    redict = pickle.load(file)

data=json.load(open('subgraph.json','r',encoding='utf-8'))

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
{e1} {r1} {e2} .
{e2} {r2} {a} .
}}'''

A7='''ASK
{{
{e1} {r1} {e2} .
{e2} {r2} {e3} .
{e3} {r3} {a} .
}}'''

A8='''ASK
{{
{e1} {r1} {e2} .
{e2} {r2} {e3} .
{e3} {r3} {e4} .
{e4} {r4} {a} .
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

corpus=[]
for sample in tqdm(data):
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
        continue
    # name for answer
    answer_name=[]
    for a in sample['answer']:
        if endict[a]['label'] is not None:
            answer_name.append(endict[a]['label'])
        else:
            valid=False
            break
    if not valid:
        continue
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
        continue   
    sample['head entity name']=head_name
    sample['answer name']=answer_name
    sample['subgraph name']=subgraph_name
    sample['relation path name']=path_name
    # for graph id 1
    if sample["graph id"]==1:
        sample['sparql']=G1.format(e1=head_name[0],r1=path_name[0])
    if sample["graph id"]==2:
        sample['sparql']=G2.format(e1=head_name[0],r1=path_name[0])
    if sample["graph id"]==3:
        sample['sparql']=G3.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1])
    if sample["graph id"]==4:
        sample['sparql']=G4.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1])        
    if sample["graph id"]==5:
        sample['sparql']=G5.format(e1=head_name[0],r1=path_name[0],e2=head_name[1],r2=path_name[1],e3=head_name[2],r3=path_name[2])
    if sample["graph id"]==6:
        sample['sparql']=G6.format(e1=head_name[0],r1=path_name[0],r2=path_name[1])
    if sample["graph id"]==7:
        sample['sparql']=G7.format(e1=head_name[0],r1=path_name[0],r2=path_name[1],r3=path_name[2])
    if sample["graph id"]==8:
        sample['sparql']=G8.format(e1=head_name[0],r1=path_name[0],r2=path_name[1],r3=path_name[2],r4=path_name[3])
    corpus.append(sample)

json.dump(corpus,open('fact_sparql.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)                    