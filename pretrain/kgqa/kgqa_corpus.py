import os
import json
import random

random.seed(42)

data=json.load(open('question_code_corpus.json','r',encoding='utf-8'))

input_prompt='''Please answer the question based on the subgraph retrieved from the knowledge graph. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all answers in a single line, starting with "Answer: " and separating each answer by "|" as follows: Answer: First answer|Second answer|Third answer ...
Subgraph: {knowledge}
Question: {ques}
'''

output_prompt='''CoT: {cot}
Answer: {answer}'''

code_corpus=[]
text_corpus=[]

for sample in data:
    # construct code corpus
    code_line=sample["code_input_output"].split('\n')
    sep=-1
    for index,line in enumerate(code_line):
        if line.strip().startswith('Answer the question based on the knowledge graph.'):
            sep=index+2
            break
    # skip this sample
    if sep==-1:
        continue
    code_dict=dict()
    code_dict["instruction"]='\n'.join(code_line[0:sep])
    code_dict["input"]=''
    code_dict["output"]='\n'.join(code_line[sep:])
    code_corpus.append(code_dict)
    
    # construct text corpus
    text_dict=dict()
    subgraph_str=''
    for t in sample["graph_extend_name"]:
        subgraph_str+=f'({t[0]}, {t[1]}, {t[2]}) '
    subgraph_str=subgraph_str.strip()
    text_dict["instruction"]=input_prompt.format(knowledge=subgraph_str,ques=sample["question"])
    text_dict["input"]=''
    text_dict["output"]=output_prompt.format(cot=sample["cot"],answer='|'.join(sample["answer name"]))
    text_corpus.append(text_dict)

# random shuffle
random.shuffle(code_corpus)
random.shuffle(text_corpus)

# save code_corpus
os.makedirs('code_corpus/train',exist_ok=True)
json.dump(code_corpus,open('code_corpus/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
TRAINING_NUM=int(len(code_corpus)*0.9)
json.dump(code_corpus[:TRAINING_NUM],open('code_corpus/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(code_corpus[TRAINING_NUM:],open('code_corpus/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

# save text_corpus
os.makedirs('text_corpus/train',exist_ok=True)
json.dump(text_corpus,open('text_corpus/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
TRAINING_NUM=int(len(text_corpus)*0.9)
json.dump(text_corpus[:TRAINING_NUM],open('text_corpus/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(text_corpus[TRAINING_NUM:],open('text_corpus/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)