import json
import random
import pickle

random.seed(42)

hop1=pickle.load(open('1hop.pkl','rb'))
endict=pickle.load(open('endict.pkl','rb'))
redict=pickle.load(open('redict.pkl','rb'))
data=json.load(open('all_question_cot.json','r',encoding='utf-8'))

data1=[]
for sample in data:
    enset=set()
    graph=set()
    for t in sample["subgraph"]:
        enset.add(t[0])
        enset.add(t[2])
        graph.add((t[0],t[1],t[2]))
    if len(graph)<30:
        # add 1-hop subgraph of head entity
        for e in sample["head entity"]:
            for t in hop1[e]:
                graph.add(t)
                if len(graph)>=30:
                    break
            if len(graph)>=30:
                break            
        # add subgraph from other entities
        for e in enset:
            for t in hop1[e]:
                graph.add(t)
                if len(graph)>=30:
                    break
            if len(graph)>=30:
                break
    graph=list(graph)
    random.shuffle(graph)
    graph_ex=[]
    graph_ex_name=[]
    for t in graph:
        graph_ex.append([t[0],t[1],t[2]])
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
        graph_ex_name.append(t_name)
    sample["extend graph"]=graph_ex
    sample["extend graph name"]=graph_ex_name
    data1.append(sample)

json.dump(data1,open('all_question_graph_extend.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)