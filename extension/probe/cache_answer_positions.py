"""Cache C_outcome hidden states at every `<answer>` opening token.

For each rollout in a list of (prompt_idx, resp_idx) pairs (the multi-answer
clean-406 rollouts identified by phase2a_per_answer_correctness.py), runs a
single forward pass and extracts the hidden state at every `<answer>` opening
token within the rollout. Saves to:

  extension/cache/probe_cache_n500_clean406/C_outcome_l<L>_answers.npz
  + .meta.json with {prompt_idx, resp_idx, answer_block_idx, tok_idx,
                     block_correct, is_first, is_last, scored_correct, n_blocks}

Designed to be run on Modal next to cache_hidden_states.py (same image).
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


_ANSWER_OPEN_RE = re.compile(r"<answer>")


def char_to_token_index(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    """Returns the first token whose offset interval contains or starts at char_idx."""
    for tok_idx, (s, e) in enumerate(offsets):
        if s <= char_idx < e or s == char_idx:
            return tok_idx
    return None


def first_token_at_or_after(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    for tok_idx, (s, _e) in enumerate(offsets):
        if s >= char_idx:
            return tok_idx
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_json", required=True,
                        help="eval_c_outcome_n500.json (contains rollouts to extract)")
    parser.add_argument("--rollouts_jsonl", required=True,
                        help="per_answer_correctness.jsonl filtered to multi-answer clean rollouts")
    parser.add_argument("--output_dir", default="/vol/probe_cache_n500_answers")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--checkpoint_name", default="C_outcome")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[answers-cache] Loading model {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    rows = [json.loads(l) for l in open(args.eval_json) if l.strip()]
    targets = [json.loads(l) for l in open(args.rollouts_jsonl) if l.strip()]
    targets = [t for t in targets if t["n_blocks"] >= 2]
    print(f"[answers-cache] {len(targets)} multi-answer rollouts to extract")

    # Cache layout
    cache = {l: {"X": [], "meta": []} for l in args.layers}

    n_processed = 0
    n_extracted = 0
    for t in targets:
        p_idx, r_idx = int(t["prompt_idx"]), int(t["resp_idx"])
        prompt_text = rows[p_idx]["prompt"]
        response = rows[p_idx]["response"][r_idx]
        scored = bool(rows[p_idx]["scores"][r_idx] == 1.0)
        per_block_correct = list(t["per_block_correct"])
        full_text = prompt_text + response

        enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True,
                        max_length=args.max_seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple of (1, T, hidden_dim) bf16

        prompt_end_char = len(prompt_text)
        # Find each '<answer>' occurrence in the response only.
        n_blocks_found = 0
        for m in _ANSWER_OPEN_RE.finditer(response):
            block_idx = n_blocks_found
            char_in_full = prompt_end_char + m.start()
            tok_idx = char_to_token_index(offsets, char_in_full)
            if tok_idx is None:
                tok_idx = first_token_at_or_after(offsets, char_in_full)
            if tok_idx is None or tok_idx >= input_ids.shape[1]:
                n_blocks_found += 1
                continue
            block_correct = per_block_correct[block_idx] if block_idx < len(per_block_correct) else None
            for layer in args.layers:
                vec = hidden_states[layer][0, tok_idx].float().cpu().numpy()
                cache[layer]["X"].append(vec)
                cache[layer]["meta"].append({
                    "prompt_idx": p_idx, "resp_idx": r_idx,
                    "answer_block_idx": block_idx, "tok_idx": int(tok_idx),
                    "block_correct": bool(block_correct) if block_correct is not None else None,
                    "is_first": block_idx == 0,
                    "is_last": block_idx == len(per_block_correct) - 1,
                    "scored_correct": scored,
                    "n_blocks": int(t["n_blocks"]),
                })
                n_extracted += 1
            n_blocks_found += 1
        n_processed += 1
        if n_processed % 200 == 0:
            print(f"[answers-cache] {n_processed}/{len(targets)} rollouts, {n_extracted} hidden states extracted")

    # Write outputs.
    for layer, data in cache.items():
        if not data["X"]:
            continue
        X = np.stack(data["X"], axis=0).astype(np.float32)
        base = f"{args.checkpoint_name}_l{layer}_answers"
        np.savez_compressed(os.path.join(args.output_dir, f"{base}.npz"), X=X)
        with open(os.path.join(args.output_dir, f"{base}.meta.json"), "w") as f:
            json.dump(data["meta"], f)
        print(f"[answers-cache] ({layer}): X={X.shape} -> {base}.npz")


if __name__ == "__main__":
    main()
