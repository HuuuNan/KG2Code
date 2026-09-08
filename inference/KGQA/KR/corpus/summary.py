import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

data=json.load(open('../../../../pretrain/kgqa/question_code_corpus.json','r',encoding='utf-8'))

client=OpenAI()

kr_prompt='''Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: (Pastura, country, United States) (Jacob Lindley, country of citizenship, United States) (Rocky Mountains, country, United States) (George Junior Republic, country, United States) (Sonny's BBQ, country, United States) (Bobby Orrock, country of citizenship, United States) (Sonny's BBQ, instance of, business) (Waste No Food, country, United States) (Whitey Moore, country of citizenship, United States) (Thomas Morrow Reavley, country of citizenship, United States) (beer in San Diego County, California, country, United States) (WNJB-FM, country, United States) (United States, country, United States) (Community Foundation for Northeast Georgia, country, United States) (Lanesville Township, country, United States) (Planet B-Boy, country of origin, United States) (Brian Dietzen, country of citizenship, United States) (Palestine, country, United States) (Jean Klock Park, country, United States) (Nampa High School, country, United States) (Nicholas Raymond Cerio, country of citizenship, United States) (Daniel Shulman, country of citizenship, United States) (John Whitaker, country of citizenship, United States) (United States, official language, English) (Adams Memorial Building, country, United States) (Mackinac Transportation Company, country, United States) (Ben Hill County Courthouse, country, United States) (Lily Lake, country, United States) (Normie Glick, country of citizenship, United States) (Sonny's BBQ, headquarters location, Winter Park)
Question: What is the official language of the country where Sonny's BBQ is located?
Knowledge: Sonny's BBQ is located in United States. The official language of United States is English.

Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: (Duke University, member of, Scholarly Publishing and Academic Resources Coalition) (Florida State University, member of, Scholarly Publishing and Academic Resources Coalition) (University of North Texas, member of, Scholarly Publishing and Academic Resources Coalition) (University of British Columbia, member of, Scholarly Publishing and Academic Resources Coalition) (Georgetown University, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Center for Research Libraries) (Wesleyan University, member of, Scholarly Publishing and Academic Resources Coalition) (University of New Mexico, member of, Scholarly Publishing and Academic Resources Coalition) (Nicolas van de Walle, educated at, Princeton University) (Ohio University, member of, Scholarly Publishing and Academic Resources Coalition) (Scholarly Publishing and Academic Resources Coalition, instance of, coalition) (Furman University, member of, Scholarly Publishing and Academic Resources Coalition) (Minnesota State University, Mankato, member of, Scholarly Publishing and Academic Resources Coalition) (Gettysburg College, member of, Scholarly Publishing and Academic Resources Coalition) (University of South Florida, member of, Scholarly Publishing and Academic Resources Coalition) (Nicolas van de Walle, instance of, human) (Rice University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Florida, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Ivy League) (Brandeis University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Connecticut, member of, Scholarly Publishing and Academic Resources Coalition) (Boston University, member of, Scholarly Publishing and Academic Resources Coalition) (Kent State University, member of, Scholarly Publishing and Academic Resources Coalition) (University of Calgary, member of, Scholarly Publishing and Academic Resources Coalition) (Michigan State University, member of, Scholarly Publishing and Academic Resources Coalition) (Princeton University, member of, Scholarly Publishing and Academic Resources Coalition) (St. Lawrence University, member of, Scholarly Publishing and Academic Resources Coalition) (Oregon State University, member of, Scholarly Publishing and Academic Resources Coalition) (Cornell University, member of, Scholarly Publishing and Academic Resources Coalition) (Temple University, member of, Scholarly Publishing and Academic Resources Coalition)
Question: How many organizations is the educational institution, where Nicolas van de Walle studied, a member of?
Knowledge: Nicolas van de Walle was educated at Princeton University. Princeton University is a member of Center for Research Libraries, Ivy League, and Scholarly Publishing and Academic Resources Coalition.

Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: (Blair, country, Australia) (Pullenvale, country, Australia) (City of Vincent, country, Australia) (Macropus robustus, endemic to, Australia) (Australian Certificate of Identity, country, Australia) (David Shepherd, country of citizenship, Australia) (Matt Moore, country of citizenship, Australia) (Mechanics' Institute, Sorrento, country, Australia) (St Kilda Football Club, country, Australia) (Rod Milgate, country of citizenship, Australia) (The Saturday Paper, country, Australia) (George Alexander Dunn, country of citizenship, Australia) (Frank Cameron Jackson, country of citizenship, Australia) (Iron Valley mine, country, Australia) (Loch railway station, country, Australia) (David Shepherd, given name, David) (Kathryn Bennetts, country of citizenship, Australia) (Mona Vale Hospital, country, Australia) (Shepherds Flat, country, Australia) (David Shepherd, member of sports team, St Kilda Football Club) (David Shepherd, instance of, human) (Concord West, country, Australia) (Gordon Rich-Phillips, country of citizenship, Australia) (Simon Target, country of citizenship, Australia) (Len Allmond, country of citizenship, Australia) (David Shepherd, family name, Shepherd) (East End Theatre District, country, Australia) (Jan Chapman, country of citizenship, Australia) (Melbourne Rectangular Stadium, country, Australia) (Rollands Plains, country, Australia)
Question: Is Geulah Cohen a member of the St Kilda Football Club, which is based in Australia?
Knowledge: David Shepherd plays for the St Kilda Football Club. St Kilda Football Club is in Australia.

Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: {triple}
Question: {ques}
Knowledge: '''

kr_prompt1='''Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: {triple}
Question: {ques}
Knowledge: '''

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
    knowledge=getResponse(kr_prompt.format(triple=triple.strip(),ques=sample["question"]))
    print(kr_prompt.format(triple=triple.strip(),ques=sample["question"]))
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
        sample['know_prompt']=kr_prompt1.format(triple=triple.strip(),ques=sample["question"])
        sample['knowledge']=knowledge.strip()
        sample['response']=answer
        process.append(sample)
        # generate 2,000 samples
        if len(process)>=4000:
            break
   
os.makedirs("summary/train", exist_ok=True)   
json.dump(process,open('summary/summary.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
    
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
    
json.dump(train,open('summary/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(dev,open('summary/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
    
    