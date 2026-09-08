import json
import re
import signal
from tqdm import tqdm

# -----------------------------
# Configs
# -----------------------------
# Dataset: LC-QuAD2.0, WikiWebQuestion
DATA = 'WikiWebQuestion'
# Model name (make sure it matches your gateway/platform)
LLM_NAME = 'gpt-5-mini'

data=json.load(open(f'{DATA}/{LLM_NAME}/answer-gpt-code-raw.json','r',encoding='utf-8'))
result_path=f'{DATA}/{LLM_NAME}/answer-gpt-code.json'

# -----------------------------
# Metrics
# -----------------------------
def metrics_cal(answer, re_answer):
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

acc = precision = recall = f1 = EM = 0
process=[]
for sample in tqdm(data):
    gold_answers=sample["answer"]
    # extract hard answer
    python_script=sample["python_script"]
    response=sample["response"]
    hard_answer=[]
    code_result=run_code(python_script)
    if code_result is not None:
        hard_answer=code_result["answer"]
    if not isinstance(hard_answer, list):
        hard_answer=str(hard_answer)
        if hard_answer.isdigit():
            hard_answer=[hard_answer]
        elif 'True' in hard_answer:
            hard_answer=['yes']
        else:
            hard_answer=['no']
    if not is_list_of_strings(hard_answer):
        hard_answer=[]
    
    # extract soft answer
    soft_answer=[]
    soft_line=''
    for line in reversed(response.split('\n')):
        if line.strip().startswith('#'):
            soft_line=line.strip()
            break
    if '[' in soft_line:
        soft_answer = re.findall(r'"([^"]+)"', soft_line)
    elif 'length' in soft_line:
        soft_answer = re.findall(r'\d+',soft_line)
    elif 'True' in soft_line:
        soft_answer=['yes']
    elif 'False' in soft_line:
        soft_answer=['no']
    if not is_list_of_strings(soft_answer):
        soft_answer=[]
        
    # use hard answer first
    if len(hard_answer)!=0 and is_list_of_strings(hard_answer):
        pred_answers=hard_answer
    else:
        pred_answers=soft_answer
    if pred_answers is None:
        pred_answers=[]
    pred_answers = [x for x in pred_answers if x is not None]
    # Metrics
    t_acc, t_prec, t_rec, t_f1, t_em = metrics_cal(gold_answers, pred_answers)
    acc += t_acc
    precision += t_prec
    recall += t_rec
    f1 += t_f1
    EM += t_em
    sample['response_answer']=pred_answers
    sample['hard_answer']=hard_answer
    sample['soft_answer']=soft_answer
    sample['accuracy']=t_acc
    sample['precision']=t_prec
    sample['recall']=t_rec
    sample['f1']=t_f1
    sample['EM']=t_em
    process.append(sample)
    
json.dump(process, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

# Final metrics
total = len(process)
print('*' * 30, 'Final Metrics', '*' * 30)
print('Accuracy: {}'.format(acc / total))
print('Precision: {}'.format(precision / total))
print('Recall: {}'.format(recall / total))
print('F1: {}'.format(f1 / total))
print('EM: {}'.format(EM / total))