import json, re, os
from typing import List, Dict, Any, Optional, Iterable, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from tqdm import tqdm  # import tqdm
from query import get_1hop
import re

DATASET = 'grailqa'
EXNUM = 100
MAX_WORKERS = os.cpu_count()  # Fix concurrency to avoid variation across machines

def extract_mids_from_sparql(query):
    # Define the regular expression pattern to match entity MIDs with different prefixes (m., n., g.)
    # The MID is prefixed with 'm.', 'n.', or 'g.' followed by alphanumeric characters
    mid_pattern = r'[mng]\.[a-zA-Z0-9_]+'

    # Use re.findall() to extract all the matches of MIDs from the SPARQL query string
    mids = re.findall(mid_pattern, query)

    return mids

def is_qid(x: str) -> bool:
    if len(x)>2 and x[:2] in ['m.','g.','n.']:
        return True
    else:
        return False

def stable_sort_triplets(triples: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    # Define deterministic sorting rule: prioritize if head/tail is QID, then use lexicographic order
    def k(t):
        h, r, t2 = t
        return (0 if is_qid(h) else 1, 0 if is_qid(t2) else 1, h, r, t2)
    return sorted(triples, key=k)

def normalize(graph):
    # Original function intended to move triples with QID tail to the front
    def tail_is_qid(g):
        return 0 if is_qid(g[-1]) else 1
    # Here we use stable sorting to guarantee reproducibility
    return sorted(graph, key=lambda g: (tail_is_qid(g), g[0], g[1], g[2]))

def top_k_unique_stable(triples: Iterable[Tuple[str,str,str]], k: int) -> List[Tuple[str,str,str]]:
    # Stable deduplication + truncation: keep the first appearance order
    seen = set()
    out = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out

def query_one_hop_stable(e: str, limit: int = 10) -> List[Tuple[str,str,str]]:
    # Wrapper for query_one_hop: apply stable sort and truncate results
    res = get_1hop(e)
    triples = [(t[0], t[1], t[2]) for t in res]
    triples = stable_sort_triplets(triples)
    return triples[:limit]

def process_one_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Collect triples in a list to preserve order, use set only for deduplication
    collected = []
    seen = set()

    def add_triple(t):
        if t not in seen:
            seen.add(t)
            collected.append(t)

    if len(sample.get("graph_mid", [])) != 0:
        # Add original graph triples (sorted first for reproducibility)
        base = []
        for g in sample["graph_mid"]:
            for t in g:
                base.append((t[0], t[1], t[2]))
        for t in stable_sort_triplets(base):
            add_triple(t)
            if len(collected) >= EXNUM:
                break
        # Extend with one-hop neighbors
        if len(collected) < EXNUM:
            candidate=[]
            for e in sample['entity']:
                if e not in sample["answer_mid"]:
                    candidate.append(e)
            for e in candidate:
                for t in query_one_hop_stable(e, limit=10):
                    add_triple(t)
                    if len(collected) >= EXNUM:
                        break
                if len(collected) >= EXNUM:
                    break
    else:
        # If no initial graph, extract QIDs from SPARQL and expand
        enmid=extract_mids_from_sparql(sample["sparql"])
        for e in enmid:
            for t in query_one_hop_stable(e, limit=10):
                add_triple(t)
                if len(collected) >= EXNUM:
                    break
            if len(collected) >= EXNUM:
                break

    # Final stable sort (optional, to enforce consistent priority)
    collected = stable_sort_triplets(collected)
    collected = top_k_unique_stable(collected, EXNUM)

    glist = [[h, r, t] for (h, r, t) in collected]
    enset, reset = set(), set()
    for h, r, t in collected:
        if is_qid(h): enset.add(h)
        if is_qid(t): enset.add(t)
        reset.add(r)

    sample["graph_extend_mid"] = glist
    sample["entity_extend"] = sorted(enset)      # Sort output for consistency
    sample["relation_extend"] = sorted(reset)
    return sample

def graph_extend_multiprocess(data, max_workers: Optional[int] = None):
    if max_workers is None:
        max_workers = MAX_WORKERS

    results = []
    total = len(data)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_sample, sample) for sample in data]
        for f in tqdm(as_completed(futures), total=total, desc="Processing (multiprocess)"):
            res = f.result()
            if res is not None:
                results.append(res)

    return results

if __name__ == "__main__":
    in_path = f'../../graph/{DATASET}/origin.json'
    out_path = f'../../graph/{DATASET}/graph.json'
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed = graph_extend_multiprocess(data)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved {len(processed)} items to {out_path}")
