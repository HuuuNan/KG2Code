import os
import sys
import json
import time
import math
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
result_path = f'{DATA}/{LLM_NAME}/answer-gpt-origin-cot.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

client=OpenAI()

# -----------------------------
# Prompt Template
# -----------------------------
prompt_tmpl = '''Please answer the question based on the subgraph retrieved from the knowledge graph. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all answers in a single line, starting with "Answer: " and separating each answer by "|" as follows: Answer: First answer|Second answer|Third answer ...
Subgraph: {knowledge}
Question: {ques}
'''

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


def parse_response_answers(response: str) -> List[str]:
    """Extract answers from the model response line starting with 'Answer:'"""
    re_answer: Set[str] = set()
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('Answer:'):
            line_ans = line.replace('Answer:', '', 1).split('|')
            for a in line_ans:
                a = a.strip()
                if a:
                    re_answer.add(a)
            break
    return list(re_answer)


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


def build_knowledge(sample: Dict[str, Any]) -> str:
    """Convert graph_extend_name into '(h, r, t)' string sequence"""
    know = ''
    if sample.get("graph_extend_name"):
        for t in sample["graph_extend_name"]:
            know += f'({t[0]}, {t[1]}, {t[2]}) '
        know = know[:-1] if know else know
    return know


def process_one(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single sample: build input -> call API -> parse -> evaluate -> return record"""
    know = build_knowledge(sample)
    inputs = prompt_tmpl.format(knowledge=know, ques=sample['question'])
    response = getResponse(inputs)

    gold_answer = sample.get("answer_name", [])
    pred_answers = parse_response_answers(response)

    temp_acc, temp_precision, temp_recall, temp_f1, temp_em = metrics_cal(gold_answer, pred_answers)

    record = {
        'index': idx,
        'question': sample['question'],
        'answer': gold_answer,
        'graph': sample.get("graph_extend_name", []),
        'input': inputs,
        'response': response,
        'response_answer': pred_answers,
        'accuracy': temp_acc,
        'precision': temp_precision,
        'recall': temp_recall,
        'f1': temp_f1,
        'EM': temp_em
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

    # Aggregate metrics
    n = len(results)
    acc_sum = sum(r['accuracy'] for r in results)
    precision_sum = sum(r['precision'] for r in results)
    recall_sum = sum(r['recall'] for r in results)
    f1_sum = sum(r['f1'] for r in results)
    em_sum = sum(r['EM'] for r in results)

    print('Accuracy: {}'.format(acc_sum / n if n else 0))
    print('Precision: {}'.format(precision_sum / n if n else 0))
    print('Recall: {}'.format(recall_sum / n if n else 0))
    print('F1: {}'.format(f1_sum / n if n else 0))
    print('EM: {}'.format(em_sum / n if n else 0))

    # Save results
    json.dump(results, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
