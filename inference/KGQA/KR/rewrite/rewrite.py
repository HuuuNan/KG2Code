import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU

import json
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Basic Configurations (same as original script)
# -----------------------------
KR = 'cotkr'                  # 'kg-to-text' | 'summary' | others (with reasoning chain)
DATASET = 'grailqa'
LLM_NAME = 'Llama-3.1-8B-Instruct'
BATCH_SIZE = 64               # batch size, adjustable based on GPU memory

# I/O
data = json.load(open(f'../../graph/{DATASET}/test.json','r',encoding='utf-8'))
result_path = f'{DATASET}/{LLM_NAME}/{KR}.json'
os.makedirs(f'{DATASET}/{LLM_NAME}', exist_ok=True)

# Model and tokenizer path (same as original script)
LLM_PATH = f'../instruction-tuning/output/{KR}/{LLM_NAME}/merge'

# -----------------------------
# Prompt Templates (same as original script)
# -----------------------------
if KR == 'kg-to-text':
    base_prompt = '''Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is: '''
elif KR == 'summary':
    base_prompt = '''Your task is to summarize the relevant knowledge that is helpful to answer the question from the following triples.
Triples: {graph}
Question: {ques}
Knowledge: '''
else:
    base_prompt = '''Your task is to summarize the relevant information that is helpful to answer the question from the following triples. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Triples: {graph}
Question: {ques}
'''

# -----------------------------
# vLLM Engine
# -----------------------------
def build_engine():
    llm = LLM(
        model=LLM_PATH,
        dtype="half",               # fp16 for efficiency
        tensor_parallel_size=1,     # single GPU
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384
    )
    return llm

def llm_response_batch(prompts, llm, sampling_params):
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
# Main: Batch Inference
# -----------------------------
def main():
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        n=1,
        max_tokens=2048,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )

    processed = []
    pbar = tqdm(total=len(data))
    for start in range(0, len(data), BATCH_SIZE):
        batch_samples = data[start:start + BATCH_SIZE]

        # ---- Build graph_str and prompt input (same logic as original script) ----
        batch_inputs = []
        for sample in batch_samples:
            graph_str = ''
            for t in sample["graph_used"]:
                graph_str = graph_str + '(' + ', '.join(t) + ') '
            graph_str = graph_str[:-1] if graph_str else ''

            if KR == 'kg-to-text':
                inputs = base_prompt.format(graph=graph_str)
            else:
                inputs = base_prompt.format(graph=graph_str, ques=sample["question"])
            token_len = len(tokenizer.encode(inputs, add_special_tokens=False))
            if token_len > llm.llm_engine.model_config.max_model_len:
                print(f"[WARN] Truncated prompt: {token_len} to {llm.llm_engine.model_config.max_model_len}")
            inputs = truncate_prompt(
                inputs,
                tokenizer,
                llm.llm_engine.model_config.max_model_len
            )
            batch_inputs.append(inputs)
        
        # ---- Run vLLM batch generation ----
        batch_outputs = llm_response_batch(batch_inputs, llm, sampling_params)
        # ---- Write results back to sample, with same field name: sample["knowledge"] ----
        for sample, knowledge in zip(batch_samples, batch_outputs):
            sample_out = dict(sample)  # avoid modifying original object
            sample_out["knowledge"] = knowledge
            processed.append(sample_out)

        pbar.update(len(batch_samples))
    pbar.close()

    # ---- Save results ----
    json.dump(processed, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved to: {result_path}  |  Total samples: {len(processed)}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
