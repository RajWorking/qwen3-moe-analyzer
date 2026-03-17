# Qwen3-30B-A3B MoE Router Analyzer

Collects token-to-expert assignment statistics and router telemetry from the official [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) Mixture of Experts model using HuggingFace Transformers.

## What it measures

| Metric | Description |
|---|---|
| **Expert activation counts** | How many times each expert is selected per layer |
| **EMA frequency** | Exponentially weighted moving average of expert selection rates |
| **Router entropy** | Shannon entropy of gate probabilities — higher means more uniform routing |
| **Top-k confidence** | Average gate probability assigned to selected experts |
| **Load-balance CV** | Coefficient of variation of expert hit counts — lower means better balance |
| **Expert co-occurrence** | Which expert pairs are frequently selected together for the same token |
| **Per-category breakdown** | When using a labeled dataset, stats are split by task category |

## Setup

```bash
pip install -r requirements.txt
```

**Hardware requirements:** Qwen3-30B-A3B in bfloat16 needs ~60 GB of GPU memory. Use `--dtype float16` or quantization for smaller footprints. Multi-GPU is supported via `--device-map auto`.

## Usage

### Dataset mode (recommended)

Run analysis over `databricks/databricks-dolly-15k` with per-category breakdowns:

```bash
python analyze-qwen3-30b-a3b.py \
    --model Qwen/Qwen3-30B-A3B \
    --dataset databricks/databricks-dolly-15k \
    --samples 50 \
    --tokens 128 \
    --output stats_30b/
```

This processes 50 samples per category (`open_qa`, `closed_qa`, `summarization`, `generation`, `classification`, `brainstorming`, `information_extraction`) and produces per-category expert activation breakdowns.

### Prompt mode

Run on ad-hoc prompts:

```bash
python analyze-qwen3-30b-a3b.py \
    --model Qwen/Qwen3-30B-A3B \
    --tokens 128 \
    --output stats_30b/ \
    "Translate to French: Hello" "Explain recursion in simple terms."
```

### Per-token tracing

Add `--trace` to log every token's expert assignment and gate scores:

```bash
python analyze-qwen3-30b-a3b.py \
    --dataset databricks/databricks-dolly-15k \
    --samples 10 \
    --trace \
    --output stats_30b/
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B` | HuggingFace model name or local path |
| `--tokens` | `128` | Max new tokens to generate per prompt |
| `--decay` | `0.99999` | EMA decay factor |
| `--output` | `stats_30b` | Output directory |
| `--dtype` | `auto` | Torch dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `--device-map` | `auto` | Device placement strategy |
| `--tp` | — | Tensor parallel size (number of GPUs) |
| `--dataset` | — | HuggingFace dataset name (enables dataset mode) |
| `--split` | `train` | Dataset split |
| `--samples` | `50` | Max samples per category (or total if no categories) |
| `--text-field` | `instruction` | Dataset column for prompt text |
| `--category-field` | `category` | Dataset column for category labels |
| `--trace` | off | Write per-token expert assignments to `token_trace.jsonl` |

## Output files

```
stats_30b/
├── layer_0.txt                  # Per-layer expert stats
├── layer_1.txt
├── ...
├── aggregated.txt               # Model-wide expert summary
├── router_telemetry.json        # Machine-readable telemetry (entropy, CV, confidence per layer)
├── token_trace.jsonl            # Per-token trace (only with --trace)
└── by_category/                 # Per-category breakdowns (only with --dataset)
    ├── brainstorming.json
    ├── classification.json
    ├── closed_qa.json
    ├── generation.json
    ├── information_extraction.json
    ├── open_qa.json
    └── summarization.json
```

## Using a different dataset

Any HuggingFace dataset works. Point `--text-field` and `--category-field` to the right columns:

```bash
# MMLU (57 academic subjects)
python analyze-qwen3-30b-a3b.py \
    --dataset cais/mmlu \
    --split test \
    --text-field question \
    --category-field subject \
    --samples 20

# Alpaca (no category labels)
python analyze-qwen3-30b-a3b.py \
    --dataset tatsu-lab/alpaca \
    --text-field instruction \
    --category-field "" \
    --samples 200
```

## How it works

The script registers PyTorch forward hooks on every `SparseMoeBlock` in the model. During each forward pass, the hook intercepts the router's gate logits, computes softmax probabilities, and records which experts are selected via top-k. Gate probabilities are used to compute entropy and confidence metrics. All statistics are accumulated globally and per-category, then written to disk at the end of the run. Hooks are cleanly removed after collection completes.
