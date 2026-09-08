import json
import pickle
from tqdm import tqdm
from query import query_ent_name
from typing import Any, Dict, List, Optional, Set
import copy

with open("prompt/base.txt", "r", encoding='utf-8') as f:
    base_prompt = f.read()

with open("prompt/kgqa.txt", "r", encoding='utf-8') as f:
    kgqa_prompt = f.read()

data=json.load(open('../../graph/webqsp/graph.json'))

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
        return ( len(node)>2 and node[:2] in ['m.','n.','g.'])

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

    return new_triples,mapping

def convert_names(triples):
    triplelist=[]
    for t in triples:
        # literal
        if t[0][0:2] not in ['m.','n.','g.']:
            head=t[0]
        # entity
        else:
            temp=query_ent_name(t[0])
            if temp:
                head=temp
            else:
                head=t[0]
        # literal
        if t[2][0:2] not in ['m.','n.','g.']:
            tail=t[2]
        # entity
        else:
            temp=query_ent_name(t[2])
            if temp:
                tail=temp
            else:
                tail=t[2]     
        triplelist.append([head,t[1],tail])
    return triplelist

# deal with dataset
process_data=[]
for sample in tqdm(data):
    answer_name=sample["answer_name"]
    if len(answer_name)==1 and answer_name[0].lower() in ['true','yes']:
        answer_name=['yes']
    if len(answer_name)==1 and answer_name[0].lower() in ['false','no']:
        answer_name=['no']
    sample["answer_name"]=answer_name
    graph_name=[]
    graph_extend_name=[]
    # get the CVT mapping of the entities
    graph_mid=[]
    for g in sample["graph_mid"]:
        for t in g:
            graph_mid.append(t)
    graph_name=convert_names(graph_mid)
    graph_name,mapping=relabel_cvt_nodes(graph_name)
    # construct the name format of graph_extend_mid
    graph_extend_name=convert_names(sample['graph_extend_mid'])
    graph_extend_name_used=copy.deepcopy(graph_extend_name)
    sample['graph_extend_name']=graph_extend_name_used
    graph_used=[]
    for t in graph_extend_name:
        if mapping.get(t[0]):
            t[0]=mapping[t[0]]
        if mapping.get(t[2]):
            t[2]=mapping[t[2]]
        graph_used.append(t)
    sample['graph_used']=graph_used
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
json.dump(process_data, open('../../graph/webqsp/test.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)