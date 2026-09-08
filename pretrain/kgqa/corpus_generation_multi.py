from openai import OpenAI
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm  # progress bar

LLM_NAME = "gpt-5-mini"
# Credentials for the OpenAI-compatible API used to generate the code corpus:
#   export OPENAI_BASE_URL="https://api.openai.com/v1"
#   export OPENAI_API_KEY="sk-..."
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ["OPENAI_API_KEY"]

# How many samples to save per checkpoint
SAVE_INTERVAL = 10000

# Multiprocessing workers
MAX_WORKERS = max(10, os.cpu_count())


def load_text(path: str) -> str:
    """Read a UTF-8 text file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_response(prompt: str, max_retries: int = 10) -> str:
    """
    Call the OpenAI API with retries.
    Retries on network errors, 429, or 5xx errors.
    """
    # Create client inside the worker process
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=LLM_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"[Worker] Error: {e}")
            # Backoff before retrying
            print("[Worker] Retrying in 60 seconds...")
            time.sleep(60)

    return ""


def build_prompt(sample: Dict[str, Any],
                 base_prompt: str,
                 kgqa_prompt: str,
                 fact_demo: str,
                 count_demo: str,
                 boolean_demo: str) -> Optional[str]:
    """Build the input prompt for a single sample. Return None if the sample should be skipped."""
    answer = sample.get("answer name", [])

    if not answer:
        return None

    # Choose demonstration prompt based on answer type
    if len(answer) == 1:
        if str(answer[0]).isdigit():
            demo = count_demo
        elif str(answer[0]) in ["yes", "no"]:
            demo = boolean_demo
        else:
            demo = fact_demo
    else:
        demo = fact_demo

    input_prompt = demo
    input_prompt += (
        "\n\nPlease complete the below code based on three demonstrations, "
        "ensuring that the format is exactly the same and the code is executable.\n\n"
        + base_prompt
    )

    # Collect entities from subgraph triples
    entity_set = set()
    for triple in sample.get("subgraph name", []):
        entity_set.add(triple[0])
        entity_set.add(triple[2])

    # Add nodes
    for entity in entity_set:
        input_prompt += f'graph.add_node("{entity}")\n'

    # Add edges
    for triple in sample.get("subgraph name", []):
        input_prompt += (
            f'graph.add_edge("{triple[0]}", "{triple[2]}", '
            f'relation="{triple[1]}")\n'
        )

    # Add question + kgqa prompt
    input_prompt += f'\nquestion = \'{sample.get("question", "")}\'\n\n'
    input_prompt += kgqa_prompt

    return input_prompt


def process_one(index_and_sample: Tuple[int, Dict[str, Any]],
                prompts: Dict[str, str]) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Process one sample: build prompt, call API, attach code_input/code_output.
    Return (index, updated_sample) or None if skipped.
    """
    idx, sample = index_and_sample

    input_prompt = build_prompt(
        sample=sample,
        base_prompt=prompts["base_prompt"],
        kgqa_prompt=prompts["kgqa_prompt"],
        fact_demo=prompts["fact_demo"],
        count_demo=prompts["count_demo"],
        boolean_demo=prompts["boolean_demo"],
    )

    if input_prompt is None:
        return None

    sample["code_input"] = input_prompt
    code_output = get_response(input_prompt)
    sample["code_output"] = code_output

    # Optional: print in worker (can be noisy in multiprocessing)
    print("*" * 30)
    print(sample["code_input"])
    print("*" * 30)
    print(sample["code_output"])

    return idx, sample


def save_checkpoint(result: list, processed_count: int) -> None:
    """Save the whole accumulated result to a checkpoint file."""
    output_file = f"all_question_code_output-{processed_count}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Checkpoint saved: {processed_count} samples -> {output_file}")


def main():
    # Load data and prompt templates in main process
    data = json.load(open("all_question_graph_extend.json", "r", encoding="utf-8"))

    prompts = {
        "base_prompt": load_text("prompt/base.txt"),
        "kgqa_prompt": load_text("prompt/kgqa.txt"),
        "fact_demo": load_text("fact_prompt.txt"),
        "count_demo": load_text("count_prompt.txt"),
        "boolean_demo": load_text("boolean_prompt.txt"),
    }

    # Store all processed samples (never cleared)
    result = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    total = len(data)

    # Submit tasks
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, (idx, sample), prompts)
                   for idx, sample in enumerate(data)]

        # Progress bar updates as futures complete
        with tqdm(total=total, desc="Processing", unit="sample") as pbar:
            for fut in as_completed(futures):
                out = fut.result()

                idx, sample_done = out
                result.append(sample_done)
                processed_count += 1

                # Save checkpoint every SAVE_INTERVAL processed samples
                if processed_count % SAVE_INTERVAL == 0:
                    save_checkpoint(result, processed_count)

                pbar.update(1)

    # Final save
    final_file = "all_question_code_output.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Final save completed: {processed_count} samples -> {final_file}")


if __name__ == "__main__":
    main()
