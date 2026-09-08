import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

random.seed(42)
data=json.load(open('count_sparql.json','r',encoding='utf-8'))
random.shuffle(data)

g_num1=[500,500,500,500,500,500,0,0]
g_num2=[6500,6500,6500,5500,5500,6500,0,0]
corpus=[]
t_num1=[0,0,0,0,0,0,0,0]
t_num2=[0,0,0,0,0,0,0,0]

fact_prompt_demo='''Please generate a corresponding question based on the SPARQL query below.
SELECT DISTINCT ?x
WHERE {{
Great American Mountain Rally country ?x .
?x public holiday Thanksgiving .
}}
Question: Which country hosts the Great American Mountain Rally and celebrates Thanksgiving as a public holiday?

Please generate a corresponding question based on the SPARQL query below.
SELECT DISTINCT ?x
WHERE {{
Landkreis Kassel shares border with e2 .
e2 applies to jurisdiction ?x .
}}
Question: Which jurisdictions does Landkreis Kassel share a border with?

Please generate a corresponding question based on the SPARQL query below.
SELECT DISTINCT ?x
WHERE {{
?x shares border with Quer .
?x country Spain .
}}
Question: Which Spanish regions share a border with Quer?

Please generate a corresponding question based on the SPARQL query below.
{sparql}
Question: 
'''

count_prompt='''Please generate a corresponding question based on the SPARQL query below.
{sparql}
Question: 
'''

# set client
client=OpenAI()

def fetch_question(sample, max_retries=100):
    sparql = sample['sparql']
    prompt = count_prompt.format(sparql=sparql)
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
            )
            question = res.choices[0].message.content.strip()
            sample['question'] = question
            return sample
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(10)
    print(f"Failed after {max_retries} retries.")
    return None

for sample in data:
    if len(sample["subgraph"])>30:
        continue
    if sample["answer"][0]=="1":
        if t_num1[sample['graph id']-1]<g_num1[sample['graph id']-1]:
            corpus.append(sample)
            t_num1[sample['graph id']-1]+=1
    else:
        if t_num2[sample['graph id']-1]<g_num2[sample['graph id']-1]:
            corpus.append(sample)
            t_num2[sample['graph id']-1]+=1        
    if all(t >= g for t, g in zip(t_num1, g_num1)) and all(t >= g for t, g in zip(t_num2, g_num2)):
        break

corpus1 = []
max_workers = 20
batch_size = 10000

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(fetch_question, sample): sample for sample in corpus}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Generating Questions"), 1):
        result = future.result()
        if result:
            corpus1.append(result)
        
        # Save every 10,000 results
        if i % batch_size == 0:
            filename = f'count_question_{i}.json'
            json.dump(corpus1, open(filename, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            print(f"Saved {len(corpus1)} questions to {filename}")
    
json.dump(corpus1,open('count_question.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
