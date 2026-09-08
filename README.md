# KG2Code

**Bridging Knowledge Graphs and Large Language Models via Executable Code for Question Answering**

[![arXiv](https://img.shields.io/badge/arXiv-2607.22652-b31b1b.svg)](https://arxiv.org/abs/2607.22652)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of **KG2Code** and the **KG2Code-QA** framework.

---

## Overview

Existing ways of combining LLMs with knowledge graphs — RAG over verbalised triples,
agent-style traversal, or SPARQL generation — either flatten the graph into text (losing its
structure), or produce reasoning that cannot be checked.

**KG2Code** instead serialises a knowledge graph as **executable Python code**, and casts KGQA
as **code completion**. The subgraph becomes a `networkx.MultiDiGraph`, the question becomes a
function signature, and the model writes the body. Running that function produces the answer,
so every reasoning step is verifiable and hallucinated hops simply fail to execute.

### The representation

A retrieved subgraph is rendered as graph-construction code (`prompt/base.txt`, then one
`add_node` / `add_edge` call per element), followed by the question and an empty `KGQA` stub
(`prompt/kgqa.txt`). That whole block is the model input:

```python
import networkx as nx

graph = nx.MultiDiGraph()

graph.add_node("Sonny's BBQ")
graph.add_node("United States")
graph.add_node("English")
graph.add_edge("Sonny's BBQ", "United States", relation="country")
graph.add_edge("United States", "English", relation="official language")

question = 'What is the official language of the country where Sonny\'s BBQ is located?'

def KGQA(question, graph):
    """
    Answer the question based on the knowledge graph.
    """
```

The fine-tuned model completes the body, interleaving code with comments that record the
intermediate result of each hop:

```python
    # Step 1: find the country of Sonny's BBQ
    country = [t for _, t, d in graph.out_edges("Sonny's BBQ", data=True)
               if d["relation"] == "country"]
    # country: ["United States"]
    # Step 2: find the official language of that country
    answer = [t for c in country for _, t, d in graph.out_edges(c, data=True)
              if d["relation"] == "official language"]
    # answer: ["English"]
    return answer
```

Evaluation appends `answer = KGQA(question, graph)` and executes the script with a 5-second
timeout — that gives the **hard answer**. If execution fails or returns nothing usable, the
last comment line is parsed instead, giving the **soft answer**. See
[`inference/KGQA/answer/answer-code.py`](inference/KGQA/answer/answer-code.py).

Three answer types are supported and handled uniformly: entity lists, counts (`len(...)`), and
booleans (`True` / `False` → `yes` / `no`).

### Pipeline

```
             Wikidata5M                     WebQSP · GrailQA · WikiWebQuestion · LC-QuAD 2.0
                  │                                              │
   ┌──────────────▼──────────────┐               ┌───────────────▼───────────────┐
   │ 1. Corpus construction      │               │ 3a. Subgraph retrieval        │
   │    pretrain/kgqa/           │               │     inference/KGQA/retrieve/  │
   │  sample subgraph            │               │  dataset → origin.json        │
   │  → templated SPARQL         │               │  → 1-hop expansion → graph.json│
   │  → LLM writes the question  │               │  → code prompt → test.json    │
   │  → LLM writes KGQA() body   │               └───────────────┬───────────────┘
   │  → execute, keep if correct │                               │
   └──────────────┬──────────────┘                               │
                  │                                              │
   ┌──────────────▼──────────────┐               ┌───────────────▼───────────────┐
   │ 2. LoRA instruction tuning  │──── merge ───▶│ 3b. Answer + evaluate         │
   │    instruction-tuning/      │               │     inference/KGQA/answer/    │
   └─────────────────────────────┘               └───────────────────────────────┘
```

Training uses **only** synthetic data built over Wikidata5M. The four benchmarks are never
trained on, so all reported numbers are **zero-shot on unseen KGs** — including Freebase, a KG
with a different schema and different entity identifiers from anything seen during tuning.

---

## Repository structure

```
KG2Code/
├── pretrain/kgqa/                    # Stage 1 — training-corpus construction (Wikidata5M)
│   ├── dict.py                       #   in/out relation & triple indices        → *.pkl
│   ├── 1hop.py                       #   1-hop neighbourhood index               → 1hop.pkl
│   ├── subgraph.py                   #   sample subgraphs over 8 query templates → subgraph.json
│   ├── subgraph2name.py              #   map QIDs/PIDs to surface labels
│   ├── fact_sparql.py                #   templated SPARQL — factoid questions
│   ├── count_sparql.py               #   templated SPARQL — counting questions
│   ├── judge_sparql.py               #   templated SPARQL — boolean questions
│   ├── fact_question.py              #   SPARQL → natural-language question (LLM, few-shot)
│   ├── count_question.py             #   ...
│   ├── judge_question.py             #   ...
│   ├── merge.py                      #   merge the three question types → all_question.json
│   ├── graph_extend.py               #   pad each subgraph to 30 triples with distractors
│   ├── corpus_generation_multi.py    #   LLM writes the KGQA() body (multiprocess + checkpoints)
│   ├── code_eval.py                  #   execute the code, keep only samples that are correct
│   ├── kgqa_corpus.py                #   → code corpus + CoT-text corpus
│   ├── prompt/{base,kgqa}.txt        #   the two prompt fragments of the representation
│   └── corpus_merge/                 # baseline / ablation corpora derived from the code corpus
│       ├── corpus_merge.py           #   merge all question families, shuffle, 90/10 split
│       ├── code2text.py              #   code corpus → plain-text CoT corpus
│       ├── gnn_corpus.py             #   corpus for the GNN baseline
│       ├── ablation-code.py          #   w/o code output   (code in, text answer out)
│       ├── ablation-comment.py       #   w/o comments      (strip the reasoning trace)
│       └── ablation-graph.py         #   w/o code input    (triple string in, code out)
│
├── instruction-tuning/               # Stage 2 — LoRA supervised fine-tuning
│   ├── run_llama.sh                  #   Llama-3.1-8B-Instruct launcher
│   ├── run_qwen.sh                   #   Qwen2.5-(Coder-)7B-Instruct launcher
│   ├── run_clm_sft_with_peft.py      #   training entry point
│   ├── build_dataset.py              #   instruction/input/output → loss-masked tensors
│   └── ds_zero2_no_offload.json      #   DeepSpeed ZeRO-2 config
│
├── inference/                        # Stage 3 — retrieval, answering, evaluation
│   ├── merge.py                      #   merge a LoRA adapter into the base model
│   └── KGQA/
│       ├── dataset/                  #   raw benchmark files (see "Datasets")
│       ├── retrieve/                 #   subgraph retrieval → code prompts
│       │   ├── webqsp/               #     Freebase; needs a local SPARQL endpoint
│       │   │   ├── retrieve.py       #       dataset → graph/webqsp/origin.json
│       │   │   ├── graph-extend-stable.py  # 1-hop expansion to ≤100 triples → graph.json
│       │   │   ├── graph-infer.py    #       CVT relabelling + code prompt → test.json
│       │   │   ├── query.py          #       SPARQL helpers
│       │   │   └── prompt/           #       base.txt, kgqa.txt
│       │   ├── grailqa/              #     same four steps for GrailQA
│       │   ├── wikiweb-*.py          #     same steps for WikiWebQuestion (Wikidata)
│       │   ├── lcquad-*.py           #     same steps for LC-QuAD 2.0 (Wikidata)
│       │   └── prompt/               #     shared by the Wikidata scripts
│       ├── answer/                   #   answering + metrics
│       │   ├── answer-code.py        #     KG2Code-QA (ours)
│       │   ├── answer-text.py        #     CoT-Tuning baseline
│       │   ├── answer-origin.py      #     untuned backbone, triples in the prompt
│       │   ├── answer-gpt-code.py    #     GPT + KG2Code, few-shot
│       │   ├── answer-gpt-code-eval.py  #  execute & score the GPT code outputs
│       │   ├── answer-gpt-cot.py     #     GPT + CoT over triples
│       │   └── answer-ablation-{code,comment,graph}.py
│       └── KR/                       #   knowledge-rewriting baselines
│           ├── corpus/               #     build KG-to-Text / Summary / CoTKR corpora
│           ├── instruction-tuning/   #     train the rewriter
│           ├── rewrite/rewrite.py    #     subgraph → rewritten knowledge
│           └── answer/answer.py      #     answer from the rewritten knowledge
│
├── vllm.txt                          # pip requirements — inference / corpus generation
└── LlamaFactory.txt                  # pip requirements — fine-tuning
```

---

## Setup

### 1. Environments

Two conda environments are used: **vllm** for corpus generation and inference,
**LlamaFactory** for fine-tuning.

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r vllm.txt

conda create -n LlamaFactory python=3.10 -y
conda activate LlamaFactory
pip install -r LlamaFactory.txt
```

Key versions: `vllm==0.12.0`, `torch==2.9.0`, `transformers==4.57.3`, `peft==0.18.0`,
`networkx==3.6.1`. Experiments were run on NVIDIA GPUs with CUDA 12.8.

### 2. Knowledge graphs

| KG | Used by | How to set up |
|---|---|---|
| **Freebase** | WebQSP, GrailQA | Follow [dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) to run a local Virtuoso endpoint. |
| **Wikidata** | WikiWebQuestion, LC-QuAD 2.0 | A local Wikidata endpoint, or the public WDQS (subject to rate limits). |
| **Wikidata5M** | Stage 1 corpus construction | Download [`wikidata5m_all_triplet.txt`](https://deepgraphlearning.github.io/project/wikidata5m) into `pretrain/wikidata5m/`. |

The Freebase endpoint is read from the environment, so you never have to edit the source:

```bash
export FREEBASE_ENDPOINT="http://localhost:3001/sparql"
```

### 3. Backbone LLMs

Download the backbones into `pretrain/`:

```
KG2Code/
└── pretrain/
    ├── Llama-3.1-8B-Instruct/          # meta-llama/Llama-3.1-8B-Instruct
    ├── Qwen2.5-7B-Instruct/            # Qwen/Qwen2.5-7B-Instruct
    └── Qwen2.5-Coder-7B-Instruct/      # Qwen/Qwen2.5-Coder-7B-Instruct
```

### 4. API credentials

Stage 1 and the GPT baselines call an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional, for a gateway
```

---

## Data and released artefacts

Preprocessed corpora and retrieval results, so you can skip the expensive stages:

| Archive | Contents | Lets you skip | Extract to |
|---|---|---|---|
| [`pretrain.zip`](https://drive.google.com/file/d/1mA8lHXNOsp5QPJjWcoqOUol67QzVF6Ye/view?usp=sharing) | KG2Code-QA training corpus (code) | Stage 1 | `instruction-tuning/` |
| [`pretrain_text.zip`](https://drive.google.com/file/d/1mY9vCZ4BSZxdJ9DblbO_2AQG82RLdULt/view?usp=sharing) | CoT-Tuning training corpus (text) | Stage 1 | `instruction-tuning/` |
| [`graph.zip`](https://drive.google.com/file/d/1zEvrBMhFoVa0K4fKXy47EL2-keVbof2L/view?usp=sharing) | retrieved subgraphs + code prompts for all four benchmarks | Stage 3a | `inference/KGQA/graph/` |
| [`corpus.zip`](https://drive.google.com/file/d/19y0CnomqBixAkw_iPWGlyrXW4KcHQ3vZ/view?usp=sharing) | KG-to-Text / Summary / CoTKR rewriting outputs | the KR baselines | `inference/KGQA/KR/rewrite/` |

### Datasets

Raw benchmark files ship in `inference/KGQA/dataset/`:

| Dataset | KG | File | Split |
|---|---|---|---|
| WebQSP | Freebase | `webqsp/test.json`, `webqsp/WebQSP.test.json` | test |
| GrailQA | Freebase | `grailqa/grailqa_v1.0_dev.json` | dev |
| WikiWebQuestion | Wikidata | `WikiWebQuestion/{train,dev,test}.json` | train / dev / test |
| LC-QuAD 2.0 | Wikidata | `LC-QuAD2.0/{test,test_with_answer}.json` | test |

---

## Usage

Each script is configured by **constants at the top of the file** (`DATA`, `LLM_NAME`,
`BATCH_SIZE`, `CUDA_VISIBLE_DEVICES`, …) rather than command-line flags. Edit those, then run
the script **from its own directory** — all paths are relative.

### Stage 1 — Build the training corpus

Skip this if you downloaded `pretrain.zip`. Run from `pretrain/kgqa/`:

```bash
python dict.py                    # relation/triple indices from wikidata5m_all_triplet.txt
python 1hop.py                    # 1-hop neighbourhood index
python subgraph.py                # sample subgraphs (8 query templates, G1–G8)
python subgraph2name.py           # QID/PID → surface labels

python fact_sparql.py   && python fact_question.py    # factoid  questions
python count_sparql.py  && python count_question.py   # counting questions
python judge_sparql.py  && python judge_question.py   # boolean  questions
python merge.py                   # → all_question.json

python graph_extend.py            # pad each subgraph to 30 triples with distractors
python corpus_generation_multi.py # LLM completes KGQA() for every sample
python code_eval.py               # execute; keep only samples whose code returns the gold answer
python kgqa_corpus.py             # → code corpus + text (CoT) corpus
```

`code_eval.py` is what makes the corpus "high-quality": generated programs are actually run,
and a sample survives only if its output matches the gold answer of the templated SPARQL query.
Nothing unverified reaches the training set.

Then, from `pretrain/kgqa/corpus_merge/`:

```bash
python corpus_merge.py            # merge question families, shuffle, 90/10 train/dev split
python code2text.py               # CoT-Tuning corpus
python ablation-code.py           # ablation corpora
python ablation-comment.py
python ablation-graph.py
```

### Stage 2 — Instruction tuning

From `instruction-tuning/`, set `llm` and `dataset` at the top of the launcher, then:

```bash
bash run_llama.sh                 # or: bash run_qwen.sh
```

LoRA SFT through DeepSpeed ZeRO-2: rank 64 / α 128 (Llama) or rank 16 / α 32 (Qwen), all
attention and MLP projections trainable, lr `1e-4` with cosine schedule, 10 epochs, max
sequence length 2048, fp16, gradient checkpointing. `dataset` selects the corpus directory
(`code_corpus`, `text_corpus`, `ablation-code`, `ablation-comment`, `ablation-graph`, …), which
must contain `train/train.json` and `dev.json`. Only the completion is supervised —
`build_dataset.py` masks the prompt tokens with `-100`.

Merge the adapter into the base model (set `LLM`, `PEFT_PATH` and `OUTPUT_PATH` first):

```bash
cd inference && python merge.py
```

### Stage 3a — Retrieve subgraphs

Skip this if you downloaded `graph.zip`.

**Freebase benchmarks** — run from `inference/KGQA/retrieve/webqsp/` (or `grailqa/`):

```bash
python retrieve.py                # dataset/  → graph/<DATA>/origin.json
python graph-extend-stable.py     # + 1-hop expansion, deterministic, ≤100 triples → graph.json
python graph-infer.py             # CVT relabelling + label lookup + code prompt → test.json
```

**Wikidata benchmarks** — run from `inference/KGQA/retrieve/`:

```bash
python wikiweb-retrieve.py    && python wikiweb-graph-extend-stable.py
python wikiweb-graph-query.py && python wikiweb-graph-infer.py

python lcquad-retrieve.py     && python lcquad-graph-extend-stable.py
python lcquad-graph-query.py  && python lcquad-graph-infer.py
```

The extra `*-graph-query.py` step batch-fetches missing entity and relation labels from
Wikidata into `endict.pkl` / `redict.pkl`.

Two details worth knowing:

- **Determinism.** `graph-extend-stable.py` sorts triples with a fixed key (QID-headed first,
  then lexicographic) and deduplicates stably, so the retrieved subgraph is identical across
  machines and worker counts.
- **CVT nodes.** Freebase compound value types have no label, so `graph-infer.py` renames them
  `CVT1`, `CVT2`, … in first-appearance order. They stay readable as code identifiers while the
  mediated structure is preserved.

### Stage 3b — Answer and evaluate

From `inference/KGQA/answer/`, set `DATA`, `LLM_NAME` and `LLM_PATH`, then:

```bash
python answer-code.py             # KG2Code-QA
```

Results land in `<DATA>/<LLM_NAME>/answer-code.json` — one record per question with the prompt,
the raw response, the executed script, the hard and soft answers, and per-sample metrics.
Accuracy / Precision / Recall / F1 / EM are printed at the end.

Metrics are computed over lower-cased, deduplicated answer sets:

| Metric | Definition |
|---|---|
| Accuracy | at least one gold answer recovered (Hits@1-style) |
| Precision / Recall / F1 | set-based, against the gold answer set |
| EM | predicted set equals the gold set exactly |

---

## Baselines and ablations

| Script | Setting |
|---|---|
| `answer/answer-origin.py` | untuned backbone, triples verbalised in the prompt |
| `answer/answer-text.py` | **CoT-Tuning** — same data, natural-language CoT instead of code |
| `answer/answer-gpt-cot.py` | GPT, few-shot CoT over triples |
| `answer/answer-gpt-code.py` + `answer-gpt-code-eval.py` | GPT, few-shot KG2Code |
| `KR/` | KG-to-Text, Summary and CoTKR knowledge rewriting, then answer |
| `answer/answer-ablation-code.py` | **w/o code output** — code input, natural-language answer |
| `answer/answer-ablation-comment.py` | **w/o comments** — code with the reasoning trace stripped |
| `answer/answer-ablation-graph.py` | **w/o code input** — triple string in, code out |

The KR baselines are a two-stage pipeline: train a rewriter (`KR/corpus/` →
`KR/instruction-tuning/`), rewrite each subgraph (`KR/rewrite/rewrite.py`, with `KR` set to
`kg-to-text`, `summary` or `cotkr`), then answer from the rewritten text (`KR/answer/answer.py`).

---

## Configuration notes

Settings you will need to change for your own machine:

| Where | What | Default |
|---|---|---|
| environment | `FREEBASE_ENDPOINT` | `http://localhost:3001/sparql` |
| environment | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | required for Stage 1 and the GPT baselines |
| every GPU script | `os.environ["CUDA_VISIBLE_DEVICES"]`, first lines of the file | `"0"` |
| `inference/merge.py` | `LLM`, `PEFT_PATH`, `OUTPUT_PATH` | — |
| `inference/KGQA/answer/*.py` | `DATA`, `LLM_NAME`, `LLM_PATH`, `BATCH_SIZE` | — |

### Known gaps

- The Wikidata scripts (`inference/KGQA/retrieve/{wikiweb,lcquad}-*.py`) import a shared
  `query.py` — `run_sparql`, `query_one_hop`, `get_label` — that is **not** in this release.
  The Freebase equivalents in `retrieve/webqsp/query.py` and `retrieve/grailqa/query.py` show
  the expected interface. Use `graph.zip` to reproduce the Wikidata results without it.
- `retrieve/{webqsp,grailqa}/query.py` optionally read a `cvt_relation.txt` list; that block is
  commented out and the scripts run without it.
- `answer/answer-gpt-code.py` expects a few-shot file `demo.txt` in its working directory.
- `pretrain/kgqa/subgraph.py` expects `entity.txt`, the pool of seed entities to sample from.

---

## Citation

```bibtex
@article{wu2026kg2code,
  title   = {KG2Code: Bridging Knowledge Graphs and Large Language Models
             via Executable Code for Question Answering},
  author  = {Wu, Yike and Hu, Nan and Qi, Guilin and Xiao, Guohui and Jiang, Chen and
             Zou, Xinchun and Lu, Yuchen and Zhai, Songlin and Chen, Yongrui and
             Zhang, Yuyang and Li, Xiaoguang and Shang, Lifeng and Chen, Jiaoyan and
             Pan, Jeff Z.},
  journal = {arXiv preprint arXiv:2607.22652},
  year    = {2026}
}
```

## Acknowledgements

This work builds on [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup),
[vLLM](https://github.com/vllm-project/vllm), [PEFT](https://github.com/huggingface/peft) and
[DeepSpeed](https://github.com/deepspeedai/DeepSpeed). We evaluate on WebQSP, GrailQA,
WikiWebQuestion and LC-QuAD 2.0, and build the training corpus over Wikidata5M.

## License

Released under the MIT License — see [LICENSE](LICENSE). The benchmark datasets and knowledge
graphs remain under their original licenses.
