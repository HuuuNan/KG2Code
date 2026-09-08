import json
import pickle
import concurrent.futures
from query import get_label
from tqdm import tqdm
from collections import defaultdict
from threading import Lock

with open("dict/endict-v30.pkl", "rb") as file:
    endict = pickle.load(file)
with open("dict/redict-v30.pkl", "rb") as file:
    redict = pickle.load(file)

data = json.load(open(f'../graph/LC-QuAD2.0/graph.json', 'r', encoding='utf-8'))

all_entities = set()
all_relations = set()
for sample in data:
    all_entities.update(sample.get("entity_extend", []))
    if sample.get("answer_mid"):
        for a in sample["answer_mid"]:
            if a[0]=='Q' and a[1:].isdigit():
                all_entities.add(a)
    all_relations.update(sample.get("relation_extend", []))

def needs_entity(e):
    if endict.get(e) is None or endict[e]["label"] is None:
        return True
    else:
        return False

def needs_relation(r):
    if redict.get(r) is None or redict[r]["label"] is None:
        return True
    else:
        return False

entities_to_fetch = [e for e in all_entities if needs_entity(e)]
relations_to_fetch = [r for r in all_relations if needs_relation(r)]

lock_e = Lock()
lock_r = Lock()

def fetch_entity(e):
    label = get_label(e)
    with lock_e:
        d = endict.get(e) or {}
        d['label'] = label
        endict[e] = d

def fetch_relation(r):
    label = get_label(r)
    with lock_r:
        d = redict.get(r) or {}
        d['label'] = label
        redict[r] = d

with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(fetch_entity, entities_to_fetch), total=len(entities_to_fetch), desc="Entities"))
    list(tqdm(executor.map(fetch_relation, relations_to_fetch), total=len(relations_to_fetch), desc="Relations"))

with open("dict/endict-v31.pkl", "wb") as f:
    pickle.dump(endict, f)
with open("dict/redict-v31.pkl", "wb") as f:
    pickle.dump(redict, f)

with open("endict.pkl", "wb") as f:
    pickle.dump(endict, f)
with open("redict.pkl", "wb") as f:
    pickle.dump(redict, f)