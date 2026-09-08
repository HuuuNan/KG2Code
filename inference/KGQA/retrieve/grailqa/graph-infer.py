import os
import json
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import copy
import re

from query import query_ent_name

def extract_quoted_content(text):
    pattern = r'["\'](.*?)["\']'
    return re.findall(pattern, text)

def load_text(path: str) -> str:
    """Load a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_mid(x: Any) -> bool:
    """Return True if x looks like a Freebase MID (m./n./g.)."""
    return isinstance(x, str) and len(x) >= 2 and x[:2] in {"m.", "n.", "g."}


def convert_names(triples: List[List[Any]]) -> List[List[Any]]:
    """
    Convert entity MID in triples to human-readable names using query_ent_name().
    Keep literals unchanged.
    """
    triple_list: List[List[Any]] = []

    for t in triples:
        # Head
        if not _is_mid(t[0]):
            head = t[0]
        else:
            temp = query_ent_name(t[0])
            head = temp if temp else t[0]

        # Tail
        if not _is_mid(t[2]):
            tail = t[2]
        else:
            temp = query_ent_name(t[2])
            tail = temp if temp else t[2]
        
        # extract content between ""
        if '"' in head:
            head = extract_quoted_content(head)[0] if extract_quoted_content(head) else head
        if '"' in tail:
            tail = extract_quoted_content(tail)[0] if extract_quoted_content(tail) else tail
        triple_list.append([head, t[1], tail])
        
    return triple_list


def build_prompt(
    base_prompt: str,
    kgqa_prompt: str,
    graph_extend_name,
    question: str,
) -> str:
    prompt = base_prompt + "\n"

    node_set = set()
    for s, _, o in graph_extend_name:
        node_set.add(str(s))
        node_set.add(str(o))

    # Add nodes
    for node in node_set:
        prompt += f"graph.add_node({repr(node)})\n"

    # Add edges
    for s, r, o in graph_extend_name:
        prompt += (
            f"graph.add_edge({repr(str(s))}, {repr(str(o))}, "
            f"relation={repr(str(r))})\n"
        )

    prompt += f"\nquestion = {repr(question)}\n\n"
    prompt += kgqa_prompt
    return prompt

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

def process_one_sample(sample: Dict[str, Any], base_prompt: str, kgqa_prompt: str) -> Dict[str, Any]:
    """
    Process one dataset sample:
      - convert graph_mid to graph_name
      - convert graph_extend_mid to graph_extend_name
      - build prompt input
    """
    out = dict(sample)  # shallow copy
    answer_name=[]
    for a in sample["answer_mid"]:
        if not _is_mid(a):
            a_name=a
        else:
            temp = query_ent_name(a)
            a_name = temp if temp else a
        answer_name.append(a_name)
    if len(answer_name)==1 and answer_name[0].lower() in ['true','yes']:
        answer_name=['yes']
    if len(answer_name)==1 and answer_name[0].lower() in ['false','no']:
        answer_name=['no']    
    out["answer_name"]=answer_name
    # Convert each subgraph in graph_mid
    graph_name: List[List[List[Any]]] = []
    for g in sample.get("graph_mid", []):
        graph_name.append(convert_names(g))
    out["graph_name"] = graph_name
    # get the CVT mapping of the entities
    graph_mid=[]
    for g in sample["graph_mid"]:
        for t in g:
            graph_mid.append(t)
    graph_name=convert_names(graph_mid)
    graph_name,mapping=relabel_cvt_nodes(graph_name)
    # Convert graph_extend_mid
    graph_extend_mid = sample.get("graph_extend_mid", [])
    graph_extend_name = convert_names(graph_extend_mid)
    graph_extend_name_used=copy.deepcopy(graph_extend_name)
    out["graph_extend_name"] = graph_extend_name_used
    graph_used=[]
    for t in graph_extend_name:
        if mapping.get(t[0]):
            t[0]=mapping[t[0]]
        if mapping.get(t[2]):
            t[2]=mapping[t[2]]
        graph_used.append(t)
    out["graph_used"]=graph_used

    # Build final prompt
    out["input"] = build_prompt(
        base_prompt=base_prompt,
        kgqa_prompt=kgqa_prompt,
        graph_extend_name=graph_used,
        question=sample.get("question", ""),
    )
    return out


def process_dataset_multiprocess(
    data: List[Dict[str, Any]],
    base_prompt: str,
    kgqa_prompt: str,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Multiprocess dataset conversion using ProcessPoolExecutor.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    results: List[Dict[str, Any]] = []
    total = len(data)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one_sample, sample, base_prompt, kgqa_prompt)
            for sample in data
        ]
        for f in tqdm(as_completed(futures), total=total, desc="Processing (multiprocess)"):
            results.append(f.result())

    return results


if __name__ == "__main__":
    base_prompt = load_text("prompt/base.txt")
    kgqa_prompt = load_text("prompt/kgqa.txt")

    data_path = "../../graph/grailqa/graph.json"
    out_path = "../../graph/grailqa/test.json"

    data = json.load(open(data_path, "r", encoding="utf-8"))
    processed = process_dataset_multiprocess(data, base_prompt, kgqa_prompt)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(processed, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} items to {out_path}")
