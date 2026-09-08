import json
import pickle
from tqdm import tqdm
from typing import List, Any, Dict

# wikidata5m endict/redict
with open("endict.pkl", "rb") as file:
    endict = pickle.load(file)

with open("redict.pkl", "rb") as file:
    redict = pickle.load(file)

with open("prompt/base.txt", "r", encoding='utf-8') as f:
    base_prompt = f.read()

with open("prompt/kgqa.txt", "r", encoding='utf-8') as f:
    kgqa_prompt = f.read()

data=json.load(open('../graph/LC-QuAD2.0/graph.json'))

def relabel_cvt_nodes(triples: List[List[Any]]) -> List[List[Any]]:
    """
    Relabel CVT nodes in a triple list into CVT1, CVT2, ... based on first-appearance order.

    CVT detection rule (as requested):
    A node is considered a CVT node if:
      - it is a string
      - len(node) > 30
      - and (node.startswith('http://www.wikidata.org/entity/statement/')
             or node.lower().startswith('q'))

    Notes:
    - Both head (t[0]) and tail (t[2]) can be CVT nodes.
    - The mapping order is determined by scanning triples in input order,
      and scanning positions t[0], t[1], t[2] left-to-right for each triple.
    - Returns (new_triples, mapping_original_to_cvt_label).
    """

    def is_cvt(node: Any) -> bool:
        """Check whether a node satisfies the CVT condition."""
        if not isinstance(node, str):
            return False
        return (
            len(node) > 30
            and (
                node.startswith("http://www.wikidata.org/entity/statement/")
                or node.lower().startswith("q")
            )
        )

    mapping: Dict[str, str] = {}
    counter = 0

    def get_label(node: str) -> str:
        """Get an existing CVT label for node, or create a new one in appearance order."""
        nonlocal counter
        if node not in mapping:
            counter += 1
            mapping[node] = f"CVT{counter}"
        return mapping[node]

    new_triples: List[List[Any]] = []
    for t in triples:
        # Keep the original triple shape; only replace nodes that are CVT.
        new_t = list(t)

        # Head entity may be CVT.
        if len(new_t) >= 1 and is_cvt(new_t[0]):
            new_t[0] = get_label(new_t[0])
            
        # Tail entity may be CVT.
        if len(new_t) >= 3 and is_cvt(new_t[2]):
            new_t[2] = get_label(new_t[2])

        new_triples.append(new_t)

    return new_triples

# deal with dataset
process_data=[]
for sample in tqdm(data):
    graph_name=[]
    # construct the name format of graph_mid
    for g in sample['graph_mid']:
        g_name=[]
        for t in g:
            t_name=[]
            if endict.get(t[0]) and endict[t[0]]['label'] is not None:
                t_name.append(endict[t[0]]['label'])
            else:
                t_name.append(t[0])
            if redict.get(t[1]) and redict[t[1]]['label'] is not None:
                t_name.append(redict[t[1]]['label'])
            else:
                t_name.append(t[1])
            if endict.get(t[2]) and endict[t[2]]['label'] is not None:
                t_name.append(endict[t[2]]['label'])
            else:
                t_name.append(t[2])
            g_name.append(t_name)
        graph_name.append(g_name)
    sample['graph_name']=graph_name 
    graph_extend_name=[]
    # construct the name format of graph_extend_mid
    for t in sample['graph_extend_mid']:
        t_name=[]
        if endict.get(t[0]) and endict[t[0]]['label'] is not None:
            t_name.append(endict[t[0]]['label'])
        else:
            t_name.append(t[0])
        if redict.get(t[1]) and redict[t[1]]['label'] is not None:
            t_name.append(redict[t[1]]['label'])
        else:
            t_name.append(t[1])
        if endict.get(t[2]) and endict[t[2]]['label'] is not None:
            t_name.append(endict[t[2]]['label'])
        else:
            t_name.append(t[2])
        graph_extend_name.append(t_name)
    sample['graph_extend_name']=graph_extend_name
    sample['graph_used']=relabel_cvt_nodes(graph_extend_name)
    # construct the name format of answer_mid
    answer_name=[]
    for a in sample["answer_mid"]:
        if endict.get(a) and endict[a]["label"] is not None:
            answer_name.append(endict[a]["label"])
        elif a.lower() in ['true','yes']:
            answer_name.append('yes')
        elif a.lower() in ['false','no']:
            answer_name.append('no')        
        else:
            answer_name.append(a)
    sample["answer_name"]=answer_name
    prompt=base_prompt+'\n'
    # collect all the entity, including literal
    enset=set()
    for t in sample['graph_used']:
        enset.add(t[0])
        enset.add(t[2])
    # deal with entity
    for e in enset:
        prompt=prompt+'graph.add_node("{name}")'.format(name=e)+'\n'
    # deal with relation
    for triple in sample['graph_used']:
        prompt += (
            f'graph.add_edge("{triple[0]}", "{triple[2]}", '
            f'relation="{triple[1]}")\n'
        )
    prompt += f'\nquestion = \'{sample["question"]}\'\n\n'
    prompt += kgqa_prompt
    # kgqa
    sample["input"]=prompt
    process_data.append(sample)

# save data after adding graph_extend_name
json.dump(process_data, open('../graph/LC-QuAD2.0/test.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)