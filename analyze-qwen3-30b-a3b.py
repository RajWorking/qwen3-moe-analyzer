#!/usr/bin/env python3
"""
analyze-qwen3-30b-a3b.py

Token-to-expert assignment statistics and router telemetry for
Qwen/Qwen3-30B-A3B Mixture of Experts model using HuggingFace Transformers.

Supports dataset-level analysis using databricks/databricks-dolly-15k (or any
HuggingFace dataset) with per-category breakdowns, as well as ad-hoc
prompts from the command line.

Collects:
  - Per-layer expert activation counts and EMA-smoothed frequencies
  - Router entropy (measures how uniformly the router distributes tokens)
  - Expert load-balance score (coefficient of variation of expert hits)
  - Top-k routing confidence (average gate probability for selected experts)
  - Per-token expert co-occurrence patterns
  - Per-category statistics when using a labeled dataset
  - Aggregated model-wide statistics with tabular output

Usage (dataset mode):
  python analyze-qwen3-30b-a3b.py \
      --model Qwen/Qwen3-30B-A3B \
      --dataset databricks/databricks-dolly-15k \
      --samples 50 \
      --tokens 128 \
      --output stats_30b/

Usage (prompt mode):
  python analyze-qwen3-30b-a3b.py \
      --model Qwen/Qwen3-30B-A3B \
      --tokens 128 \
      --output stats_30b/ \
      "Translate to French: Hello" "Explain recursion."

Requirements: pip install torch transformers accelerate tabulate numpy datasets
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

# Force unbuffered stdout so progress is visible when redirected to a file
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tabulate import tabulate
except ImportError:
    print("Please install tabulate: pip install tabulate")
    exit(1)

# ---------------------------------------------------------------------------
# Global statistics collectors
# ---------------------------------------------------------------------------
router_hits = defaultdict(Counter)
ema_stats = defaultdict(lambda: defaultdict(float))
gate_entropy_records = defaultdict(list)
gate_confidence_records = defaultdict(list)
expert_cooccurrence = defaultdict(lambda: defaultdict(int))
token_expert_log = []

# Per-category stats (for dataset mode)
category_router_hits = defaultdict(lambda: defaultdict(Counter))
category_ema_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
category_gate_entropy = defaultdict(lambda: defaultdict(list))
category_gate_confidence = defaultdict(lambda: defaultdict(list))

current_decay = 0.99999
current_category = None
enable_token_log = False


# ---------------------------------------------------------------------------
# MoE forward hook — lightweight GPU-resident buffering
# ---------------------------------------------------------------------------
# Per-prompt buffers: layer_idx -> list of (topk_indices_cpu, topk_scores_cpu, gate_probs_cpu)
_prompt_buffers = defaultdict(list)


def make_router_hook(layer_idx):
    """Create a forward hook for the Qwen3MoeTopKRouter at the given layer index.

    The hook only detaches and stashes tensors — all stats are computed in
    flush_prompt_buffers() after the entire prompt completes.
    """

    def hook_fn(module, args, output):
        gate_probs, topk_scores, topk_indices = output
        # Detach and move to CPU immediately to avoid cross-device issues
        # during flush (layers may live on different GPUs with device_map=auto)
        _prompt_buffers[layer_idx].append((
            topk_indices.detach().cpu(),
            topk_scores.detach().float().cpu(),
            gate_probs.detach().float().cpu(),
        ))

    return hook_fn


def flush_prompt_buffers():
    """Process all buffered hook data for the current prompt, then clear buffers."""
    for layer_idx, entries in _prompt_buffers.items():
        if not entries:
            continue

        # Concatenate all forward-pass captures for this layer
        all_indices = torch.cat([e[0].reshape(-1, e[0].shape[-1]) for e in entries], dim=0)
        all_scores = torch.cat([e[1].reshape(-1, e[1].shape[-1]) for e in entries], dim=0)
        all_gate_probs = torch.cat([e[2].reshape(-1, e[2].shape[-1]) for e in entries], dim=0)

        k = all_indices.shape[-1]
        inds_flat = all_indices.reshape(-1)
        total_events = inds_flat.numel()

        if total_events == 0:
            continue

        num_experts = all_gate_probs.shape[-1]

        # Expert hit counts (already on CPU)
        bin_counts = torch.bincount(inds_flat, minlength=num_experts)

        # Entropy (CPU) — guard against nan from bfloat16 precision loss
        gate_2d = all_gate_probs.nan_to_num(nan=0.0).clamp(min=1e-12)
        ent_per_token = -(gate_2d * gate_2d.log2()).sum(dim=-1)
        entropy = ent_per_token.mean().item()

        # Confidence — guard against nan
        scores_clean = all_scores.nan_to_num(nan=0.0)
        mean_conf = scores_clean.mean().item()

        for exp in range(num_experts):
            cnt = bin_counts[exp].item()
            if cnt == 0:
                continue
            pct = cnt / total_events
            ema_stats[layer_idx][exp] = (
                ema_stats[layer_idx][exp] * current_decay + pct * (1 - current_decay)
            )
            router_hits[layer_idx][exp] += cnt

            if current_category is not None:
                cat = current_category
                category_ema_stats[cat][layer_idx][exp] = (
                    category_ema_stats[cat][layer_idx][exp] * current_decay
                    + pct * (1 - current_decay)
                )
                category_router_hits[cat][layer_idx][exp] += cnt

        gate_entropy_records[layer_idx].append(entropy)
        gate_confidence_records[layer_idx].append(mean_conf)

        if current_category is not None:
            category_gate_entropy[current_category][layer_idx].append(entropy)
            category_gate_confidence[current_category][layer_idx].append(mean_conf)

        # Co-occurrence (already on CPU)
        if k > 1:
            inds_2d_cpu = all_indices.tolist()
            for token_experts in inds_2d_cpu:
                token_experts_sorted = sorted(token_experts)
                for i in range(len(token_experts_sorted)):
                    for j in range(i + 1, len(token_experts_sorted)):
                        pair = (token_experts_sorted[i], token_experts_sorted[j])
                        expert_cooccurrence[layer_idx][pair] += 1

        # Per-token trace
        if enable_token_log:
            inds_2d_cpu = all_indices.tolist()
            scores_2d_cpu = all_scores.tolist()
            for tok_experts, tok_scores in zip(inds_2d_cpu, scores_2d_cpu):
                token_expert_log.append({
                    "layer": layer_idx,
                    "category": current_category,
                    "experts": [int(e) for e in tok_experts],
                    "scores": [round(float(s), 6) for s in tok_scores],
                })

    _prompt_buffers.clear()


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------
def _layer_telemetry_row(layer_idx):
    counts = router_hits[layer_idx]
    all_experts = sorted(set(counts.keys()) | set(ema_stats.get(layer_idx, {}).keys()))
    hit_vals = [counts.get(e, 0) for e in all_experts] if all_experts else []
    avg_ent = float(np.mean(gate_entropy_records[layer_idx])) if gate_entropy_records[layer_idx] else 0.0
    avg_conf = float(np.mean(gate_confidence_records[layer_idx])) if gate_confidence_records[layer_idx] else 0.0
    cv = float(np.std(hit_vals) / np.mean(hit_vals)) if hit_vals and np.mean(hit_vals) > 0 else 0.0
    num_active = sum(1 for v in hit_vals if v > 0)
    return {
        "all_experts": all_experts,
        "hit_vals": hit_vals,
        "avg_entropy": avg_ent,
        "avg_confidence": avg_conf,
        "load_cv": cv,
        "num_active": num_active,
    }


def save_and_print_stats(output_dir, decay_val):
    os.makedirs(output_dir, exist_ok=True)

    if not router_hits:
        print("[INFO] No router hits recorded.")
        return

    total_selections = 0
    model_expert_hits = Counter()
    model_expert_active_layers = defaultdict(int)
    model_expert_ema_sum = defaultdict(float)

    sorted_layers = sorted(router_hits.keys())

    # ── Per-layer stats ──────────────────────────────────────────────────
    print("\n--- Per-Layer Expert Activation Statistics ---")
    for layer_idx in sorted_layers:
        counts = router_hits[layer_idx]
        layer_ema = ema_stats.get(layer_idx, {})
        layer_selections = sum(counts.values())
        total_selections += layer_selections

        info = _layer_telemetry_row(layer_idx)

        table_rows = []
        sorted_by_ema = sorted(info["all_experts"], key=lambda e: -layer_ema.get(e, 0.0))
        for exp in sorted_by_ema:
            ema_val = layer_ema.get(exp, 0.0)
            hits = counts.get(exp, 0)
            table_rows.append([exp, f"{ema_val * 100:.2f}%", hits])
            model_expert_hits[exp] += hits
            if hits > 0 or ema_val > 0:
                model_expert_active_layers[exp] += 1
                model_expert_ema_sum[exp] += ema_val

        print(f"\nLayer {layer_idx}  (selections={layer_selections}, "
              f"entropy={info['avg_entropy']:.3f} bits, confidence={info['avg_confidence']:.4f}, "
              f"load-CV={info['load_cv']:.3f}, decay={decay_val})")
        if table_rows:
            print(tabulate(table_rows, headers=["Expert", "EMA (%)", "Hits"], tablefmt="grid"))

        fname = os.path.join(output_dir, f"layer_{layer_idx}.txt")
        with open(fname, "w") as f:
            f.write(f"Layer {layer_idx}, EMA(decay={decay_val})\n")
            f.write(f"Token Selections: {layer_selections}\n")
            f.write(f"Router Entropy (avg): {info['avg_entropy']:.4f} bits\n")
            f.write(f"Top-k Confidence (avg): {info['avg_confidence']:.6f}\n")
            f.write(f"Load-Balance CV: {info['load_cv']:.4f}\n\n")
            for exp in sorted_by_ema:
                f.write(f"Expert {exp}: EMA={layer_ema.get(exp, 0.0) * 100:.2f}%  Hits={counts.get(exp, 0)}\n")
        print(f"  -> {fname}")

    # ── Router telemetry summary ─────────────────────────────────────────
    print("\n\n--- Router Telemetry Summary ---")
    telemetry_rows = []
    for layer_idx in sorted_layers:
        info = _layer_telemetry_row(layer_idx)
        telemetry_rows.append([
            layer_idx, info["num_active"],
            f"{info['avg_entropy']:.3f}", f"{info['avg_confidence']:.4f}",
            f"{info['load_cv']:.3f}", sum(router_hits[layer_idx].values()),
        ])
    print(tabulate(
        telemetry_rows,
        headers=["Layer", "Active Experts", "Avg Entropy (bits)", "Avg Confidence", "Load CV", "Selections"],
        tablefmt="grid",
    ))

    # ── Expert co-occurrence ─────────────────────────────────────────────
    if expert_cooccurrence:
        print("\n\n--- Top Expert Co-occurrence Pairs (all layers) ---")
        global_pairs = Counter()
        for pair_counts in expert_cooccurrence.values():
            for pair, cnt in pair_counts.items():
                global_pairs[pair] += cnt
        top_pairs = global_pairs.most_common(20)
        pair_rows = [[f"({a}, {b})", cnt] for (a, b), cnt in top_pairs]
        print(tabulate(pair_rows, headers=["Expert Pair", "Co-occurrences"], tablefmt="grid"))

    # ── Per-category summary (dataset mode) ──────────────────────────────
    if category_router_hits:
        print("\n\n--- Per-Category Expert Activation Summary ---")
        cat_summary_rows = []
        for cat in sorted(category_router_hits.keys()):
            cat_total = sum(
                sum(layer_ctr.values())
                for layer_ctr in category_router_hits[cat].values()
            )
            cat_expert_totals = Counter()
            for layer_ctr in category_router_hits[cat].values():
                cat_expert_totals.update(layer_ctr)
            top3 = cat_expert_totals.most_common(3)
            top3_str = ", ".join(f"E{e}({c})" for e, c in top3)

            cat_entropies = []
            for layer_id in category_gate_entropy[cat]:
                cat_entropies.extend(category_gate_entropy[cat][layer_id])
            avg_cat_ent = float(np.mean(cat_entropies)) if cat_entropies else 0.0

            cat_confidences = []
            for layer_id in category_gate_confidence[cat]:
                cat_confidences.extend(category_gate_confidence[cat][layer_id])
            avg_cat_conf = float(np.mean(cat_confidences)) if cat_confidences else 0.0

            cat_summary_rows.append([
                cat, cat_total, f"{avg_cat_ent:.3f}", f"{avg_cat_conf:.4f}", top3_str,
            ])

        print(tabulate(
            cat_summary_rows,
            headers=["Category", "Total Selections", "Avg Entropy", "Avg Confidence", "Top-3 Experts"],
            tablefmt="grid",
        ))

        cat_dir = os.path.join(output_dir, "by_category")
        os.makedirs(cat_dir, exist_ok=True)
        for cat in sorted(category_router_hits.keys()):
            cat_path = os.path.join(cat_dir, f"{cat}.json")
            cat_data = {}
            for layer_id in sorted(category_router_hits[cat].keys()):
                layer_ctr = category_router_hits[cat][layer_id]
                layer_ema = category_ema_stats[cat].get(layer_id, {})
                all_exp = sorted(set(layer_ctr.keys()) | set(layer_ema.keys()))
                cat_data[str(layer_id)] = {
                    "expert_hits": {str(e): layer_ctr.get(e, 0) for e in all_exp},
                    "expert_ema": {str(e): round(layer_ema.get(e, 0.0), 8) for e in all_exp},
                    "total_selections": sum(layer_ctr.values()),
                }
            with open(cat_path, "w") as f:
                json.dump(cat_data, f, indent=2)
        print(f"[INFO] Saved per-category stats -> {cat_dir}/")

    # ── Aggregated model-wide stats ──────────────────────────────────────
    print("\n\n--- Aggregated Model-Wide Expert Statistics ---")
    all_ids = sorted(set(model_expert_hits.keys()) | set(model_expert_active_layers.keys()))
    sorted_by_hits = sorted(all_ids, key=lambda e: -model_expert_hits.get(e, 0))

    agg_rows = []
    for exp in sorted_by_hits:
        hits = model_expert_hits.get(exp, 0)
        n_layers = model_expert_active_layers.get(exp, 0)
        avg_ema = (model_expert_ema_sum.get(exp, 0.0) / n_layers * 100) if n_layers > 0 else 0.0
        sel_pct = (hits / total_selections * 100) if total_selections > 0 else 0.0
        agg_rows.append([exp, f"{avg_ema:.2f}%", n_layers, hits, f"{sel_pct:.2f}%"])

    agg_headers = ["Expert", "Avg EMA (%)", "Active Layers", "Total Hits", "Selection (%)"]
    print(f"\nTotal Token Selections (all MoE layers): {total_selections}")
    if agg_rows:
        print(tabulate(agg_rows, headers=agg_headers, tablefmt="grid"))

    agg_path = os.path.join(output_dir, "aggregated.txt")
    with open(agg_path, "w") as f:
        f.write(f"Aggregated Expert Statistics (decay={decay_val})\n")
        f.write(f"Total Token Selections: {total_selections}\n\n")
        f.write(tabulate(agg_rows, headers=agg_headers, tablefmt="pipe"))
        f.write("\n")
    print(f"\n[INFO] Saved aggregated stats -> {agg_path}")

    telemetry_path = os.path.join(output_dir, "router_telemetry.json")
    telemetry_data = {}
    for layer_idx in sorted_layers:
        counts = router_hits[layer_idx]
        all_experts = sorted(set(counts.keys()) | set(ema_stats.get(layer_idx, {}).keys()))
        hit_vals = [counts.get(e, 0) for e in all_experts] if all_experts else []
        telemetry_data[str(layer_idx)] = {
            "avg_entropy": float(np.mean(gate_entropy_records[layer_idx])) if gate_entropy_records[layer_idx] else 0.0,
            "avg_confidence": float(np.mean(gate_confidence_records[layer_idx])) if gate_confidence_records[layer_idx] else 0.0,
            "load_balance_cv": float(np.std(hit_vals) / np.mean(hit_vals)) if hit_vals and np.mean(hit_vals) > 0 else 0.0,
            "total_selections": sum(counts.values()),
            "expert_hits": {str(e): counts.get(e, 0) for e in all_experts},
            "expert_ema": {str(e): round(ema_stats[layer_idx].get(e, 0.0), 8) for e in all_experts},
        }
    with open(telemetry_path, "w") as f:
        json.dump(telemetry_data, f, indent=2)
    print(f"[INFO] Saved router telemetry -> {telemetry_path}")

    if enable_token_log and token_expert_log:
        trace_path = os.path.join(output_dir, "token_trace.jsonl")
        with open(trace_path, "w") as f:
            for entry in token_expert_log:
                f.write(json.dumps(entry) + "\n")
        print(f"[INFO] Saved token-level trace ({len(token_expert_log)} entries) -> {trace_path}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset_prompts(dataset_name, split, samples_per_category, text_field, category_field):
    """Load prompts from a HuggingFace dataset, optionally grouped by category."""
    from datasets import load_dataset as hf_load

    print(f"[INFO] Loading dataset: {dataset_name} (split={split})")
    ds = hf_load(dataset_name, split=split)

    has_categories = category_field and category_field in ds.column_names
    if not has_categories:
        if category_field:
            print(f"[WARNING] Category field '{category_field}' not found. Available: {ds.column_names}")
        print("[INFO] Loading without category grouping.")
        prompts_out = []
        for row in ds:
            text = row.get(text_field, "")
            if text and text.strip():
                prompts_out.append((text.strip(), None))
                if samples_per_category and len(prompts_out) >= samples_per_category:
                    break
        return prompts_out

    categories = sorted(set(ds[category_field]))
    print(f"[INFO] Found {len(categories)} categories: {categories}")

    prompts_out = []
    for cat in categories:
        cat_rows = [r for r in ds if r[category_field] == cat]
        n = min(len(cat_rows), samples_per_category) if samples_per_category else len(cat_rows)
        for row in cat_rows[:n]:
            text = row.get(text_field, "")
            if text and text.strip():
                prompts_out.append((text.strip(), cat))

    print(f"[INFO] Loaded {len(prompts_out)} prompts across {len(categories)} categories.")
    return prompts_out


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------
def collect_stats(model_name, prompts_with_categories, decay, output_dir,
                  torch_dtype_str, device_map, tp_size):
    global current_decay, current_category
    current_decay = decay

    # Resolve dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, "auto")

    print(f"[INFO] Loading model: {model_name}  (dtype={torch_dtype_str}, device_map={device_map})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if tp_size and tp_size > 1:
        model_kwargs["device_map"] = "auto"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(tp_size)))

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    print("[INFO] Model loaded.")

    # --- Register hooks on MoE blocks ---
    hook_handles = []
    moe_count = 0

    # Walk model to find MoE router (gate) modules inside SparseMoeBlocks.
    # We hook the router directly so we can read its output tuple
    # (gate_probs, topk_scores, topk_indices) without re-calling it.
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if "TopKRouter" in class_name:
            # Extract layer index from module path, e.g. "model.layers.5.mlp.gate"
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                layer_idx = moe_count  # fallback

            handle = module.register_forward_hook(make_router_hook(layer_idx))
            hook_handles.append(handle)
            moe_count += 1

    if moe_count == 0:
        print("[ERROR] No MoE blocks found. Ensure the model is a Qwen3 MoE variant.")
        print("[DEBUG] Model module names:")
        for name, _ in model.named_modules():
            print(f"  {name}")
        return

    print(f"[INFO] Registered hooks on {moe_count} MoE blocks.")
    print(f"[INFO] Processing {len(prompts_with_categories)} prompts...")

    from tqdm import tqdm
    pbar = tqdm(prompts_with_categories, desc="Processing", unit="prompt", file=sys.stderr)
    for prompt_text, category in pbar:
        current_category = category
        cat_label = f" [{category}]" if category else ""
        pbar.set_postfix_str(f"{cat_label} {prompt_text[:40]}...", refresh=True)

        if tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt_text
        else:
            formatted = prompt_text

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        # With device_map="auto", model is sharded — send inputs to the
        # device of the first parameter (the embedding layer's device).
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        with torch.no_grad():
            # Single forward pass (prefill only) — captures all routing
            # decisions for the input tokens without slow autoregressive decoding.
            _ = model(**inputs)

        flush_prompt_buffers()

    current_category = None

    # Remove hooks
    for h in hook_handles:
        h.remove()
    print(f"[INFO] Removed {len(hook_handles)} hooks.")

    save_and_print_stats(output_dir, decay)


def main():
    parser = argparse.ArgumentParser(
        description="Token-to-expert assignment stats & router telemetry for Qwen3-30B-A3B"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name or path (default: Qwen/Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--decay", type=float, default=0.99999,
        help="EMA decay factor"
    )
    parser.add_argument(
        "--output", default="stats_30b",
        help="Output directory for stats and telemetry files"
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Enable per-token expert assignment trace (writes token_trace.jsonl)"
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading (default: auto)"
    )
    parser.add_argument(
        "--device-map", default="auto",
        help="Device map for model placement (default: auto)"
    )
    parser.add_argument(
        "--tp", type=int, default=None,
        help="Tensor parallel size (number of GPUs)"
    )

    # Dataset options
    dataset_group = parser.add_argument_group("dataset options")
    dataset_group.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset name (e.g., databricks/databricks-dolly-15k)"
    )
    dataset_group.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: train)"
    )
    dataset_group.add_argument(
        "--samples", type=int, default=50,
        help="Max samples per category (dataset mode) or total samples (no categories). Default: 50"
    )
    dataset_group.add_argument(
        "--text-field", type=str, default="instruction",
        help="Dataset column containing the prompt text (default: instruction)"
    )
    dataset_group.add_argument(
        "--category-field", type=str, default="category",
        help="Dataset column containing the category label (default: category)"
    )

    parser.add_argument(
        "prompts", nargs="*",
        help="Prompts to run (ignored if --dataset is set)"
    )
    args = parser.parse_args()

    global enable_token_log
    enable_token_log = args.trace

    if args.dataset:
        prompts_with_cats = load_dataset_prompts(
            args.dataset, args.split, args.samples,
            args.text_field, args.category_field,
        )
    elif args.prompts:
        prompts_with_cats = [(p, None) for p in args.prompts]
    else:
        parser.error("Provide either --dataset or positional prompts.")
        return

    collect_stats(
        args.model, prompts_with_cats, args.decay, args.output,
        args.dtype, args.device_map, args.tp,
    )


if __name__ == "__main__":
    main()
