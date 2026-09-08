import json
import os

code_train=json.load(open('code_corpus/train/train.json','r',encoding='utf-8'))
code_dev=json.load(open('code_corpus/dev.json','r',encoding='utf-8'))

text_train=json.load(open('text_corpus/train/train.json','r',encoding='utf-8'))
text_dev=json.load(open('text_corpus/dev.json','r',encoding='utf-8'))

train=[]
for c_sample,t_sample in zip(code_train,text_train):
    sample=dict()
    sample["instruction"]=c_sample["instruction"].split('def KGQA(question, graph):')[0].strip()
    sample["input"]=""
    sample["output"]=t_sample["output"]
    train.append(sample)

dev=[]
for c_sample,t_sample in zip(code_dev,text_dev):
    sample=dict()
    sample["instruction"]=c_sample["instruction"].split('def KGQA(question, graph):')[0].strip()
    sample["input"]=""
    sample["output"]=t_sample["output"]
    dev.append(sample)
 
os.makedirs('ablation-code/train',exist_ok=True)   
json.dump(train,open('ablation-code/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open('ablation-code/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)