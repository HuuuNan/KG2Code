import json
import os

code_train=json.load(open('code_corpus/train/train.json','r',encoding='utf-8'))
code_dev=json.load(open('code_corpus/dev.json','r',encoding='utf-8'))

train=[]
for sample in code_train:
    temp=dict()
    temp["instruction"]=sample["instruction"].split('"""')[0].strip()
    temp["input"]=""
    output=[]
    for line in sample["output"].split('\n'):
        if line.strip().startswith('#'):
            continue
        output.append(line)
    temp["output"]='\n'.join(output)
    train.append(temp)

dev=[]
for sample in code_dev:
    temp=dict()
    temp["instruction"]=sample["instruction"].split('"""')[0].strip()
    temp["input"]=""
    output=[]
    for line in sample["output"].split('\n'):
        if line.strip().startswith('#'):
            continue
        output.append(line)
    temp["output"]='\n'.join(output)
    dev.append(temp)

os.makedirs('ablation-comment/train',exist_ok=True)   
json.dump(train,open('ablation-comment/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open('ablation-comment/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)