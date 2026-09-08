import json
import re
import os
from tqdm import tqdm

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
    
def extract_question(text):
    """
    Extract the question from the provided string.
    Assumes the question comes after the 'question = ' part of the string.
    """
    # Use a regular expression to extract everything after 'question = '
    match = re.search(r'question\s*=\s*\'(.*)\'', text)
    
    # If a match is found, return the question text
    if match:
        return match.group(1)
    else:
        return None  # Return None if no question is found

def extract_entities(comment_line: str):
    match = re.search(r":\s*(.+)", comment_line)
    if not match:
        return []
    entities = [e.strip() for e in match.group(1).split(",") if e.strip()]
    return entities
    
def extract_answer(code):
    # extract soft answer
    soft_answer=[]
    soft_line=''
    for line in reversed(code.split('\n')):
        if line.strip().startswith('#'):
            soft_line=line.strip()
            break
    if '[' in soft_line:
        soft_answer = re.findall(r'"([^"]+)"', soft_line)
    elif 'length' in soft_line:
        soft_answer = re.findall(r'\d+',soft_line)
    elif 'True' in soft_line:
        soft_answer=['yes']
    elif 'False' in soft_line:
        soft_answer=['no']
    return soft_answer

index=0
data=json.load(open('code_corpus/train/train.json','r',encoding='utf-8'))
corpus=[]
for sample in tqdm(data):
    t_list=extract_triplets(sample["instruction"])
    ques=extract_question(sample["instruction"])
    head=[]
    for line in sample["output"].split('\n'):
        if line.strip().startswith('# head entity:'):
            head=extract_entities(line)
            break
    answer=extract_answer(sample["output"])
    temp=dict()
    temp['id']=str(index)
    temp["question"]=ques
    temp["answer"]=answer
    temp["q_entity"]=head
    temp["a_entity"]=answer
    temp["graph"]=t_list
    temp["choices"]=[]
    corpus.append(temp)
    index+=1
    
os.makedirs('gnn',exist_ok=True)
with open("gnn/train.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

data=json.load(open('code_corpus/dev.json','r',encoding='utf-8'))
corpus=[]
for sample in tqdm(data):
    t_list=extract_triplets(sample["instruction"])
    ques=extract_question(sample["instruction"])
    head=[]
    for line in sample["output"].split('\n'):
        if line.strip().startswith('# head entity:'):
            head=extract_entities(line)
            break
    answer=extract_answer(sample["output"])
    temp=dict()
    temp['id']=str(index)
    temp["question"]=ques
    temp["answer"]=answer
    temp["q_entity"]=head
    temp["a_entity"]=answer
    temp["graph"]=t_list
    temp["choices"]=[]
    corpus.append(temp)
    index+=1

os.makedirs('gnn',exist_ok=True)
with open("gnn/dev.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")