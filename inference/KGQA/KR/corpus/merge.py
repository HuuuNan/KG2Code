import json
import random
import os
random.seed(42)

KR='cotkr'

data1=json.load(open(KR+'/kgqa.json','r',encoding='utf-8'))
data2=json.load(open(KR+'/kgc.json','r',encoding='utf-8'))

merged=data1+data2
random.shuffle(merged)

train_num=int(len(merged)*0.9)
train=[]
dev=[]
for i in merged[:train_num]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    train.append(temp)
    
for i in merged[train_num:]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    dev.append(temp)
 
os.makedirs(KR+"/train", exist_ok=True)   
json.dump(train,open(KR+'/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open(KR+'/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
