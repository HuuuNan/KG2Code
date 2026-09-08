import os
import json
from typing import Dict, Any, Optional, List, Set, Tuple
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

process=[]
data=json.load(open('../../dataset/webqsp/test.json','r',encoding='utf-8'))
for sample in data:
    process_sample=dict()
    process_sample["question"] = sample["question"]
    process_sample["answer_mid"] = sample["answermid"].split('|')
    if len(sample["answername"])==0:
        process_sample["answer_name"]=sample["answermid"].split('|')
    else:
        process_sample["answer_name"]=sample["answername"].split('|')
    process_sample["graph_mid"] = sample["graphmid"]
    process_sample["graph_name"]=sample["graph"]
    enset=set()
    reset=set()
    for g in sample["graphmid"]:
        for t in g:
            if t[0][:2] in ['m.','n.','g.']:
                enset.add(t[0])
            if t[2][:2] in ['m.','n.','g.']:
                enset.add(t[2])
            reset.add(t[1])
    process_sample["entity"] = list(enset)
    process_sample["relation"] = list(reset)
    process.append(process_sample)

os.makedirs('../../graph/webqsp', exist_ok=True)
out_path = '../../graph/webqsp/origin.json'
json.dump(process, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f"Saved {len(process)} items to {out_path}")
