"""Cache hidden states at EVERY `</think>` token in each rollout.

Existing `cache_hidden_states.py` extracts only the FIRST `</think>` per
rollout. Multi-answer C_outcome rollouts at 0.5B emit ~7.6 `</think>` tokens
per rollout (one before each `<answer>` block, since the model alternates
`<think>...</think> <answer>...</answer>` cycles).

This script extracts at every `</think>` with metadata:
    prompt_idx, resp_idx, think_close_idx (0 = first, 1 = second, ...),
    tok_idx, total_think_closes_in_rollout, scored_correct

Output: extension/cache/probe_cache_n500_all_thinkclose/<ckpt>_l<L>_thinkclose.npz
+ sidecar meta JSON.

Cost: ~10 min on H100 for ~6500 rollouts × 7.6 avg positions × 3 layers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_THINK_CLOSE_RE = re.compile(r"</think>")


def char_to_token_index(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    for tok_idx, (s, e) in enumerate(offsets):
        if s <= char_idx < e or s == char_idx:
            return tok_idx
    return None


def first_token_at_or_after(offsets, char_idx: int) -> int | None:
    for tok_idx, (s, _e) in enumerate(offsets):
        if s >= char_idx:
            return tok_idx
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--checkpoint_name", required=True)
    parser.add_argument("--output_dir", default="/vol/probe_cache_n500_all_thinkclose")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--max_responses_per_prompt", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[thinkclose-cache] loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    rows = [json.loads(l) for l in open(args.eval_json) if l.strip()]
    print(f"[thinkclose-cache] {len(rows)} prompts", flush=True)

    cache = {l: {"X": [], "meta": []} for l in args.layers}
    n_rollouts_processed = 0
    n_positions_total = 0

    for p_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        responses = row["response"][: args.max_responses_per_prompt]
        scores = row["scores"][: args.max_responses_per_prompt]
        prompt_end_char = len(prompt_text)

        for r_idx, (response, score) in enumerate(zip(responses, scores)):
            n_rollouts_processed += 1
            full_text = prompt_text + response
            enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True,
                            max_length=args.max_seq_len, return_tensors="pt")
            input_ids = enc["input_ids"].to("cuda")
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]

            # Find all </think> occurrences in the response only
            think_close_positions = []
            for m in _THINK_CLOSE_RE.finditer(response):
                char_in_full = prompt_end_char + m.start()
                tok_idx = char_to_token_index(offsets, char_in_full)
                if tok_idx is None:
                    tok_idx = first_token_at_or_after(offsets, char_in_full)
                if tok_idx is not None and tok_idx < input_ids.shape[1]:
                    think_close_positions.append(tok_idx)

            if not think_close_positions:
                continue

            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states

            n_total = len(think_close_positions)
            for tc_idx, tok_idx in enumerate(think_close_positions):
                for layer in args.layers:
                    vec = hidden_states[layer][0, tok_idx].float().cpu().numpy()
                    cache[layer]["X"].append(vec)
                    cache[layer]["meta"].append({
                        "prompt_idx": p_idx,
                        "resp_idx": r_idx,
                        "think_close_idx": tc_idx,
                        "tok_idx": int(tok_idx),
                        "total_think_closes": n_total,
                        "is_first": tc_idx == 0,
                        "is_last": tc_idx == n_total - 1,
                        "scored_correct": bool(float(score) == 1.0),
                    })
                    n_positions_total += 1
        if (p_idx + 1) % 50 == 0:
            print(f"[thinkclose-cache] {p_idx + 1}/{len(rows)} prompts, "
                  f"{n_rollouts_processed} rollouts, {n_positions_total} positions", flush=True)

    for layer, data in cache.items():
        if not data["X"]:
            continue
        X = np.stack(data["X"], axis=0).astype(np.float32)
        base = f"{args.checkpoint_name}_l{layer}_thinkclose"
        np.savez_compressed(os.path.join(args.output_dir, f"{base}.npz"), X=X)
        with open(os.path.join(args.output_dir, f"{base}.meta.json"), "w") as f:
            json.dump(data["meta"], f)
        print(f"[thinkclose-cache] ({layer}): X={X.shape} -> {base}.npz", flush=True)


if __name__ == "__main__":
    main()
