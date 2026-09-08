import json
import re
import signal
from tqdm import tqdm

data=json.load(open('all_question_code_output.json','r',encoding='utf-8'))

with open("prompt/base.txt", "r", encoding='utf-8') as f:
    base_prompt = f.read()

with open("prompt/kgqa.txt", "r", encoding='utf-8') as f:
    kgqa_prompt = f.read()

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

corpus=[]
for sample in tqdm(data):
    answer=sample["answer name"]
    # extract code function
    code_function=''
    if sample["code_output"] is None:
        continue
    output_line=sample["code_output"].split('\n')
    start_id=0
    for index,line in enumerate(output_line):
        if line.startswith('def KGQA(question, graph):'):
            start_id=index
    code_function='\n'.join(output_line[start_id:])
    # construct the code sample
    code_all=''
    code_all=code_all+base_prompt+'\n'
    entity_set = set()
    for triple in sample["extend graph name"]:
        entity_set.add(triple[0])
        entity_set.add(triple[2])
    for entity in entity_set:
        code_all += f'graph.add_node("{entity}")\n'
    for triple in sample["extend graph name"]:
        code_all += (
            f'graph.add_edge("{triple[0]}", "{triple[2]}", '
            f'relation="{triple[1]}")\n'
        )
    code_all += f'\nquestion = \'{sample["question"]}\'\n\n'
    code_all += code_function
    # construct the complete python script
    python_all=''
    python_all=code_all+'\nanswer = KGQA(question, graph)'
    # soft answer
    soft_answer=[]
    soft_line=''
    for line in reversed(output_line):
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
    print(soft_answer)
    # hard answer
    hard_answer=[]
    code_result=run_code(python_all)
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
    print(hard_answer)
    # judge soft answer and hard answer is correct or not (exact match)
    answer_lower = {a.lower() for a in answer}
    # make sure the format is correct
    if is_list_of_strings(soft_answer) and is_list_of_strings(hard_answer):
        soft_lower = {a.lower() for a in soft_answer}
        hard_lower = {a.lower() for a in hard_answer}
    else:
        soft_lower=set()
        hard_lower=set()
    # collect this sample
    if soft_lower==answer_lower and hard_lower==answer_lower:
        sample['code_input_output']=code_all
        sample['python_script']=python_all
        sample['soft_answer']=soft_answer
        sample['hard_answer']=hard_answer
        corpus.append(sample)

json.dump(corpus,open('question_code_corpus.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print(len(data))
print(len(corpus))
        