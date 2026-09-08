import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU device

import json
import sys
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Basic Configurations
# -----------------------------
KR = 'kg-to-text'                  # 'kg-to-text' | 'summary' | others (with reasoning chain)
# WikiWebQuestion, LC-QuAD2.0
DATASET = 'webqsp'
REWRITE = 'Llama-3.1-8B-Instruct'
# LLM_NAME: Llama-3.1-8B-Instruct, Qwen2.5-Coder-7B-Instruct
LLM_NAME = 'Qwen2.5-Coder-7B-Instruct'
BATCH_SIZE = 64  # Batch size (adjust based on GPU memory)

# I/O paths
test = json.load(open(f'../rewrite/{DATASET}/{REWRITE}/{KR}.json', 'r', encoding='utf-8'))
result_path = f'{DATASET}/{REWRITE}/{LLM_NAME}/{KR}.json'
os.makedirs(f'{DATASET}/{REWRITE}/{LLM_NAME}', exist_ok=True)
LLM_PATH = f'../../../../../pretrain/{LLM_NAME}'

# -----------------------------
# Prompt Template
# -----------------------------
prompt_tmpl = '''Please answer the question based on the following knowledge. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all answers in a single line, starting with "Answer: " and separating each answer by "|" as follows: Answer: First answer|Second answer|Third answer ...
Knowledge: {knowledge}
Question: {ques}
'''

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

def build_engine():
    """Build vLLM engine. The tokenizer (for EOS ID) is retrieved later via llm.get_tokenizer()."""
    llm = LLM(
        model=LLM_PATH,
        dtype="half",                # fp16 for speed and memory efficiency
        tensor_parallel_size=1,      # Single GPU (controlled by CUDA_VISIBLE_DEVICES)
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )
    return llm

# -----------------------------
# Batch Inference with vLLM
# -----------------------------
def llm_response_batch(prompts, llm, sampling_params):
    """Run vLLM inference on a batch of prompts and return generated texts."""
    results = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    texts = []
    for r in results:
        texts.append(r.outputs[0].text.strip() if r.outputs else "")
    return texts

# -----------------------------
# Main (Batch Inference)
# -----------------------------
def main():
    # Build engine and tokenizer
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    # Sampling configuration (equivalent to original GenerationConfig)
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

        # Build prompts from knowledge and question
        batch_inputs = []
        for s in batch_samples:
            know = s.get('knowledge', '')
            ques = s.get('question', '')
            prompt=prompt_tmpl.format(knowledge=know, ques=ques)
            messages=[{"role": "user", "content": prompt}]
            messages_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )            
            batch_inputs.append(messages_template)

        # Run inference
        batch_responses = llm_response_batch(batch_inputs, llm, sampling_params)

        # Parse responses and compute metrics
        for sample, prompt_str, response in zip(batch_samples, batch_inputs, batch_responses):
            index += 1
            gold_answers = sample.get("answer_name", [])

            # Extract answers from the line starting with "Answer:"
            pred_answers = set()
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Answer:'):
                    for a in line.replace('Answer:', '').split('|'):
                        a = a.strip()
                        if a:
                            pred_answers.add(a)
                    break
            pred_answers = list(pred_answers)

            # Metrics
            t_acc, t_prec, t_rec, t_f1, t_em = metrics_cal(gold_answers, pred_answers)
            acc += t_acc
            precision += t_prec
            recall += t_rec
            f1 += t_f1
            EM += t_em

            # Record per-sample result
            data.append({
                'question': sample.get('question', ''),
                'answer': gold_answers,
                'graph': sample.get("graph_extend_name"),
                'prompt': prompt_str,
                'response': response,
                'response_answer': pred_answers,
                'accuracy': t_acc,
                'precision': t_prec,
                'recall': t_rec,
                'f1': t_f1,
                'EM': t_em
            })

            # Streaming logs
            print(prompt_str)
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
    print('Accuracy: {}'.format(acc / total if total else 0))
    print('Precision: {}'.format(precision / total if total else 0))
    print('Recall: {}'.format(recall / total if total else 0))
    print('F1: {}'.format(f1 / total if total else 0))
    print('EM: {}'.format(EM / total if total else 0))

    # Save results
    json.dump(data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
