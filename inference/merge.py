from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM,LlamaTokenizer
from peft import PeftModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Llama-3.1-8B-Instruct, Qwen2.5-Coder-7B-Instruct, Qwen2.5-7B-Instruct
LLM='Qwen2.5-7B-Instruct'
LLM_PATH = '../../pretrain/' + LLM
# path for tokenizer
TOKENIZER_PATH = '../../pretrain/' + LLM
# path for lora
PEFT_PATH = 'checkpoint-text/' + LLM + '/checkpoint-4371'
OUTPUT_PATH='checkpoint-text/' + LLM + '/merge'

if 'Llama-2' not in LLM:
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cuda:0'
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
else:
    llm = LlamaForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cuda:0'
    )    
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

llm.resize_token_embeddings(len(tokenizer))

llm = PeftModel.from_pretrained(
    llm,
    PEFT_PATH,
    torch_dtype=torch.float16,
    device_map='cuda:0'
)
llm=llm.merge_and_unload()
llm.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
