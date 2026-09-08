import json
import os
import re

code_train=json.load(open('code_corpus/train/train.json','r',encoding='utf-8'))
code_dev=json.load(open('code_corpus/dev.json','r',encoding='utf-8'))

def extract_triplets(code_str):
    """
    Extract all (h, r, t) triplets from the provided code string.
    
    Parameters:
    - code_str: The string of code from which to extract the triplets.
    
    Returns:
    - A list of tuples representing the triplets (h, r, t).
    """
    
    # Regular expression to match the add_edge method calls
    edge_pattern = r'graph\.add_edge\("([^"]+)", "([^"]+)", relation="([^"]+)"\)'
    
    # Find all matches in the code string
    matches = re.findall(edge_pattern, code_str)
    
    # Create the list of triplets (h, r, t)
    triplets = [[h, r, t] for h, t, r in matches]
    
    return triplets
 
train=[]
for sample in code_train:
    graph=extract_triplets(sample["instruction"])
    g_str=''
    for t in graph:
        g_str+=f"({t[0]}, {t[1]}, {t[2]}) "
    g_str=g_str.strip()
    ques_part='question = '+sample["instruction"].split('question = ')[-1].strip()
    temp=dict()
    temp["instruction"]="graph = '"+g_str+'"\n\n'+ques_part
    temp["input"]=""
    temp["output"]=sample["output"]
    train.append(temp)
    
dev=[]
for sample in code_dev:
    graph=extract_triplets(sample["instruction"])
    g_str=''
    for t in graph:
        g_str+=f"({t[0]}, {t[1]}, {t[2]}) "
    g_str=g_str.strip()
    ques_part='question = '+sample["instruction"].split('question = ')[-1].strip()
    temp=dict()
    temp["instruction"]="graph = '"+g_str+"'\n\n"+ques_part
    temp["input"]=""
    temp["output"]=sample["output"]
    dev.append(temp)
    
os.makedirs('ablation-graph/train',exist_ok=True)   
json.dump(train,open('ablation-graph/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open('ablation-graph/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)