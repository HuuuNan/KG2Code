import json
import math
import copy
import time
import random
from openai import OpenAI
from tqdm import tqdm
import os

data=json.load(open('../../../../pretrain/kgqa/question_code_corpus.json','r',encoding='utf-8'))

client=OpenAI()

kr_prompt='''Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: (Pastura, country, United States) (Jacob Lindley, country of citizenship, United States) (Rocky Mountains, country, United States) (George Junior Republic, country, United States) (Sonny's BBQ, country, United States) (Bobby Orrock, country of citizenship, United States) (Sonny's BBQ, instance of, business) (Waste No Food, country, United States) (Whitey Moore, country of citizenship, United States) (Thomas Morrow Reavley, country of citizenship, United States) (beer in San Diego County, California, country, United States) (WNJB-FM, country, United States) (United States, country, United States) (Community Foundation for Northeast Georgia, country, United States) (Lanesville Township, country, United States) (Planet B-Boy, country of origin, United States) (Brian Dietzen, country of citizenship, United States) (Palestine, country, United States) (Jean Klock Park, country, United States) (Nampa High School, country, United States) (Nicholas Raymond Cerio, country of citizenship, United States) (Daniel Shulman, country of citizenship, United States) (John Whitaker, country of citizenship, United States) (United States, official language, English) (Adams Memorial Building, country, United States) (Mackinac Transportation Company, country, United States) (Ben Hill County Courthouse, country, United States) (Lily Lake, country, United States) (Normie Glick, country of citizenship, United States) (Sonny's BBQ, headquarters location, Winter Park). The sentence is: The United States is home to a variety of entities and individuals, all connected by their association with the country. Notable people such as Jacob Lindley, Bobby Orrock, Whitey Moore, Thomas Morrow Reavley, Brian Dietzen, Nicholas Raymond Cerio, Daniel Shulman, John Whitaker, and Normie Glick hold citizenship in the United States. The country is also the location for numerous places and organizations, including the Rocky Mountains, George Junior Republic, Jean Klock Park, Nampa High School, Adams Memorial Building, Mackinac Transportation Company, Ben Hill County Courthouse, Lily Lake, and Lanesville Township. Businesses like Sonny's BBQ, which is headquartered in Winter Park, and initiatives such as Waste No Food and the Community Foundation for Northeast Georgia, also operate within the United States. Additionally, the country is the origin of cultural works like "Planet B-Boy," and it recognizes English as its official language. 

Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: (Duke University, member of, Scholarly Publishing and Academic Resources Coalition) (Florida State University, member of, Scholarly Publishing and Academic Resources Coalition) (University of North Texas, member of, Scholarly Publishing and Academic Resources Coalition) (University of British Columbia, member of, Scholarly Publishing and Academic Resources Coalition) (Georgetown University, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Center for Research Libraries) (Wesleyan University, member of, Scholarly Publishing and Academic Resources Coalition) (University of New Mexico, member of, Scholarly Publishing and Academic Resources Coalition) (Nicolas van de Walle, educated at, Princeton University) (Ohio University, member of, Scholarly Publishing and Academic Resources Coalition) (Scholarly Publishing and Academic Resources Coalition, instance of, coalition) (Furman University, member of, Scholarly Publishing and Academic Resources Coalition) (Minnesota State University, Mankato, member of, Scholarly Publishing and Academic Resources Coalition) (Gettysburg College, member of, Scholarly Publishing and Academic Resources Coalition) (University of South Florida, member of, Scholarly Publishing and Academic Resources Coalition) (Nicolas van de Walle, instance of, human) (Rice University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Florida, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Ivy League) (Brandeis University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Connecticut, member of, Scholarly Publishing and Academic Resources Coalition) (Boston University, member of, Scholarly Publishing and Academic Resources Coalition) (Kent State University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Calgary, member of, Scholarly Publishing and Academic Resources Coalition) (Michigan State University, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Scholarly Publishing and Academic Resources Coalition) (St. Lawrence University, member of, Scholarly Publishing and Academic Resources Coalition) (Oregon State University, member of, Scholarly Publishing and Academic Resources Coalition) (Cornell University, member of, Scholarly Publishing and Academic Resources Coalition) (Temple University, member of, Scholarly Publishing and Academic Resources Coalition). The sentence is: The Scholarly Publishing and Academic Resources Coalition (SPARC) is a coalition whose members include Duke University, Florida State University, University of North Texas, University of British Columbia, Georgetown University, Wesleyan University, University of New Mexico, Ohio University, Furman University, Minnesota State University, Mankato, Gettysburg College, University of South Florida, Rice University, University of Florida, Brandeis University, University of Connecticut, Boston University, Kent State University, University of Calgary, Michigan State University, Princeton University, St. Lawrence University, Oregon State University, Cornell University, and Temple University. Princeton University is also a member of the Ivy League and of the Center for Research Libraries, and Nicolas van de Walle, a human, was educated at Princeton University.

Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: (Blair, country, Australia) (Pullenvale, country, Australia) (City of Vincent, country, Australia) (Macropus robustus, endemic to, Australia) (Australian Certificate of Identity, country, Australia) (David Shepherd, country of citizenship, Australia) (Matt Moore, country of citizenship, Australia) (Mechanics' Institute, Sorrento, country, Australia) (St Kilda Football Club, country, Australia) (Rod Milgate, country of citizenship, Australia) (The Saturday Paper, country, Australia) (George Alexander Dunn, country of citizenship, Australia) (Frank Cameron Jackson, country of citizenship, Australia) (Iron Valley mine, country, Australia) (Loch railway station, country, Australia) (David Shepherd, given name, David) (Kathryn Bennetts, country of citizenship, Australia) (Mona Vale Hospital, country, Australia) (Shepherds Flat, country, Australia) (David Shepherd, member of sports team, St Kilda Football Club) (David Shepherd, instance of, human) (Concord West, country, Australia) (Gordon Rich-Phillips, country of citizenship, Australia) (Simon Target, country of citizenship, Australia) (Len Allmond, country of citizenship, Australia) (David Shepherd, family name, Shepherd) (East End Theatre District, country, Australia) (Jan Chapman, country of citizenship, Australia) (Melbourne Rectangular Stadium, country, Australia) (Rollands Plains, country, Australia). The sentence is: Australia is a country that hosts a diverse array of places, individuals, and entities. Notable citizens such as David Shepherd, Matt Moore, Rod Milgate, George Alexander Dunn, Frank Cameron Jackson, Kathryn Bennetts, Gordon Rich-Phillips, Simon Target, Len Allmond, and Jan Chapman hold Australian citizenship. The country is home to various locations and organizations, including Blair, Pullenvale, City of Vincent, Mechanics' Institute in Sorrento, St Kilda Football Club, The Saturday Paper, Iron Valley mine, Loch railway station, Mona Vale Hospital, Shepherds Flat, Concord West, East End Theatre District, Melbourne Rectangular Stadium, and Rollands Plains. Additionally, Australia is the endemic region for species like Macropus robustus and is associated with documents such as the Australian Certificate of Identity. David Shepherd, a human with the given name David and family name Shepherd, is also a member of the St Kilda Football Club.

Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is: '''

kr_prompt1='Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is: '

ans_prompt='''Below are the facts that might be relevant to answer the question:
{knowledge}
Question: {ques}
Answer:'''

num_dict = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

def getResponse(prompt,max_retries=10):
    # set retries
    retries=0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model='gpt-5-mini',
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 1 minutes...")
            retries += 1
            time.sleep(60)
    return ''
    
process=[]
for sample in tqdm(data):
    # triples linearization
    triple=''
    for t in sample["extend graph name"]:
        triple=triple+'('+t[0]+', '+t[1]+', '+t[2]+') '
    triple=triple.strip()
    knowledge=getResponse(kr_prompt.format(graph=triple.strip()))
    print(kr_prompt.format(graph=triple.strip()))
    print(knowledge)
    if knowledge is None:
        continue
    answer=getResponse(ans_prompt.format(knowledge=knowledge.strip(),ques=sample["question"]))
    print(ans_prompt.format(knowledge=knowledge.strip(),ques=sample["question"]))
    print(answer)
    if answer is None:
        continue
    gold=sample["answer name"]
    # gold number
    gold_num=[]
    for i in gold:
        if i.isdigit() and num_dict.get(i):
            gold_num.append(num_dict[i])
    FLAG=False
    # judge use gold_num or gold
    for i in gold_num:
        if i.lower() in answer.lower():
            FLAG=True
            break
    if FLAG:
        gold_ans=gold_num
    else:
        gold_ans=gold

    # result
    result=''
    FLAG=True
    for i in gold_ans:
        if i.lower() not in answer.lower():
            FLAG=False
            break
    
    # record
    if FLAG:
        sample['know_prompt']=kr_prompt1.format(graph=triple.strip())
        sample['knowledge']=knowledge.strip()
        sample['response']=answer
        process.append(sample)
        # generate 4,000 samples
        if len(process)>=4000:
            break
   
os.makedirs("kg-to-text/train", exist_ok=True)   
json.dump(process,open('kg-to-text/kg-to-text.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

# convert data to specific template
train_num=int(len(process)*0.9)
train=[]
dev=[]
for i in process[:train_num]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    train.append(temp)
    
for i in process[train_num:]:
    temp=dict()
    temp["instruction"]=i['know_prompt']
    temp["input"]=''
    temp["output"]=i['knowledge']
    dev.append(temp)
    
json.dump(train,open('kg-to-text/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open('kg-to-text/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)