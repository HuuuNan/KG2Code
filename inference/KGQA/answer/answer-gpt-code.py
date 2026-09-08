import os
import sys
import json
import time
import math
import signal
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Set

from openai import OpenAI
from openai._exceptions import APIStatusError, RateLimitError, APIConnectionError

# -----------------------------
# Configs
# -----------------------------
# Dataset: QALD-9, QALD-10
DATA = 'webqsp'
# Model name (make sure it matches your gateway/platform)
LLM_NAME = 'gpt-5-mini'

# I/O paths
test = json.load(open(f'../graph/{DATA}/test.json', 'r', encoding='utf-8'))
result_path = f'{DATA}/{LLM_NAME}/answer-gpt-code-raw.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

client=OpenAI()

# -----------------------------
# Prompt Template
# -----------------------------
with open('demo.txt','r',encoding='utf-8') as f:
    demo=f.read().strip()

class Timeout(Exception):
    pass

def handler(signum, frame):
    raise Timeout()

def run_code(code: str, timeout=5):
    env = {}
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        exec(code, env)
        return env
    except Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)

def is_list_of_strings(obj):
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)

# -----------------------------
# Metrics
# -----------------------------
def metrics_cal(answer: List[str], re_answer: List[str]) -> Tuple[float, float, float, float, float]:
    """Compute Accuracy, Precision, Recall, F1, EM (case-insensitive, deduplicated)."""
    answer = list(set(a.lower() for a in answer))
    re_answer = list(set(a.lower() for a in re_answer))

    if not answer and not re_answer:
        return 1, 1, 1, 1, 1

    cor = 0
    acc_FLAG = False
    em_FLAG = True

    if answer:
        for a in answer:
            if a in re_answer:
                acc_FLAG = True
                cor += 1
            else:
                em_FLAG = False

        acc = 1 if acc_FLAG else 0
        precision = cor / len(re_answer) if re_answer else 0
        recall = cor / len(answer)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        em = 1 if em_FLAG and len(answer) == len(re_answer) else 0
    else:
        acc = precision = recall = f1 = em = 0

    return acc, precision, recall, f1, em

def getResponse(prompt: str, max_retries: int = 10) -> str:
    """API call with exponential backoff. Retries on 429/5xx/network issues."""
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=LLM_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
            )
            if res.choices[0].message.content is None:
                continue
            return res.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 1 minutes...")
            time.sleep(60)
    return ""

def process_one(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single sample: build input -> call API -> parse -> evaluate -> return record"""
    prompt=demo+'\n\n'+sample["input"]
    response = getResponse(prompt)
    response_part=response.split('"""')[-1].strip()
    python_script=sample["input"]+'\n    '+response_part+'\nanswer = KGQA(question, graph)'

    record = {
        'index': idx,
        'question': sample['question'],
        'answer': sample.get("answer_name", []),
        'graph': sample.get("graph_used", []),
        'input': sample["input"],
        'response': response,
        'python_script': python_script
    }
    return record

def main():
    MAX_WORKERS = os.cpu_count()

    futures = []
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, sample in enumerate(test, start=1):
            fut = executor.submit(process_one, idx, sample)
            futures.append(fut)

        # Progress bar updates as tasks complete
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            rec = fut.result()
            results.append(rec)

    # Sort results by index to keep order consistent
    results.sort(key=lambda x: x['index'])

    # Save results
    json.dump(results, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()