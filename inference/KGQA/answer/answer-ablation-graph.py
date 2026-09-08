import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # force GPU id=2 for vLLM

import signal
import json
import re
import sys
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Configs
# -----------------------------
# QALD-9, QALD-10, MintQA-Pop, KQAPro, WikiWebQuestion, LC-QuAD2.0, SimpleQuestionsWikidata, webqsp, MetaQA, grailqa
DATA = 'webqsp'
# Meta-Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct,deepseek-coder-7b-instruct-v1.5, Seed-Coder-8B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-Coder-7B-Instruct
LLM_NAME = 'Llama-3.1-8B-Instruct'
BATCH_SIZE = 64  # <-- explicit batch size knob
# I/O paths
test = json.load(open(f'../graph/{DATA}/test.json', 'r', encoding='utf-8'))
result_path = f'{DATA}/{LLM_NAME}/ablation-graph.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

# Local model paths
LLM_PATH = f'../../ablation-graph/{LLM_NAME}/merge'  # base model path

prompt_tmpl='''graph = '{knowledge}'

question = '{ques}'

def KGQA(question, graph):
    """
    Answer the question based on the knowledge graph.
    """'''

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

# -----------------------------
# vLLM Model Loader
# -----------------------------
def build_engine():
    """
    Build vLLM engine and (optionally) enable a LoRA adapter.
    We retrieve the tokenizer later via llm.get_tokenizer() for EOS id.
    """
    llm = LLM(
        model=LLM_PATH,
        dtype="half",                # fp16 for speed/memory
        tensor_parallel_size=1,      # single visible GPU (id=2)
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=16384
        #max_model_len=4096
    )
    return llm


# -----------------------------
# Batch Inference via vLLM
# -----------------------------
def llm_response_batch(prompts, llm, sampling_params):
    """
    Generate model responses for a batch of prompts using vLLM.
    Attach LoRA via LoRARequest only when PEFT_PATH is provided.
    """
    results = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    texts = []
    for r in results:
        texts.append(r.outputs[0].text.strip() if r.outputs else "")
    return texts

def truncate_prompt(prompt: str, tokenizer, max_model_len: int):
    """
    Truncate prompt by token length to max_model_len.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_model_len:
        input_ids = input_ids[:max_model_len-100]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt

# -----------------------------
# Main (Batch Inference)
# -----------------------------
def main():
    # Build engine and tokenizer
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    # vLLM sampling configuration (aligned with prior scripts)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        n=1,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    index = 0
    acc = precision = recall = f1 = EM = 0
    data = []

    pbar = tqdm(total=len(test))
    for start in range(0, len(test), BATCH_SIZE):
        batch_samples = test[start:start + BATCH_SIZE]
        batch_inputs = []
        for s in batch_samples:
            know = ''
            if s.get("graph_used"):
                triples = []
                for t in s["graph_used"]:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        triples.append(f"({t[0]}, {t[1]}, {t[2]})")
                know = ' '.join(triples)
            prompt=prompt_tmpl.format(knowledge=know, ques=s['question'])
            token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_len > llm.llm_engine.model_config.max_model_len:
                print(f"[WARN] Truncated prompt: {token_len} to {llm.llm_engine.model_config.max_model_len}")
            prompt = truncate_prompt(
                prompt,
                tokenizer,
                llm.llm_engine.model_config.max_model_len
            )
            batch_inputs.append(prompt)

        # Run vLLM generation
        batch_responses = llm_response_batch(batch_inputs, llm, sampling_params)

        # Parse responses and compute metrics
        for sample, prompt_str, response in zip(batch_samples, batch_inputs, batch_responses):
            index += 1
            gold_answers = sample["answer_name"]
            
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
            pred_answers=soft_answer
            pred_answers = [x for x in pred_answers if x is not None]
            # Metrics
            t_acc, t_prec, t_rec, t_f1, t_em = metrics_cal(gold_answers, pred_answers)
            acc += t_acc
            precision += t_prec
            recall += t_rec
            f1 += t_f1
            EM += t_em

            # Record per-sample data
            data.append({
                'question': sample['question'],
                'input': sample['input'],
                'answer': gold_answers,
                'graph': sample.get("graph_used"),
                'prompt': prompt_str,
                'response': response,
                'response_answer': pred_answers,
                'soft_answer': soft_answer,
                'accuracy': t_acc,
                'precision': t_prec,
                'recall': t_rec,
                'f1': t_f1,
                'EM': t_em,
                #'sparql': sample['sparql']
            })

            # Streaming logs
            #print(batch_inputs)
            print(response)
            print('Current Accuracy: {}'.format(acc / index))
            print('Current Precision: {}'.format(precision / index))
            print('Current Recall: {}'.format(recall / index))
            print('Current F1: {}'.format(f1 / index))
            print('Current EM: {}'.format(EM / index))
            sys.stdout.flush()

        pbar.update(len(batch_samples))
    pbar.close()

    # Final metrics
    total = len(test)
    print('*' * 30, 'Final Metrics', '*' * 30)
    print('Accuracy: {}'.format(acc / total))
    print('Precision: {}'.format(precision / total))
    print('Recall: {}'.format(recall / total))
    print('F1: {}'.format(f1 / total))
    print('EM: {}'.format(EM / total))

    # Persist results
    json.dump(data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
