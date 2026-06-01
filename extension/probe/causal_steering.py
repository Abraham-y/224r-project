"""Causal steering at the </think> token, layer 16, on C_outcome.

For each prompt+rollout in a test set:
  1. Take the prompt + everything up to and including the first `</think>`
     in the response (the "prefix").
  2. Re-generate the completion (from </think> onwards) under each of these
     conditions:
       - alpha=0           (baseline, no patch -- should reproduce original behavior in distribution)
       - alpha=+k * h_mean (push residual at </think> in the +probe direction; correctness UP)
       - alpha=-k * h_mean (push in -probe direction; correctness DOWN)
       - alpha=+k * h_mean along a RANDOM unit direction (control: same magnitude)
     for k in {0.5, 1.0, 2.0}.
  3. Score the new completion's final <answer> against the Countdown verifier.

Patch is applied as a forward hook on Qwen2's `model.layers[L].forward`
output for the `</think>` token position. Because the K/V cache for that
position is filled from the modified post-layer-L residual, every later
token attending to </think> sees the modified representation.

Outputs JSONL with one row per (prompt, resp_idx, condition).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# defer heavy imports until needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def find_think_close_char(prompt_text: str, response: str) -> int:
    """Return char offset of the first `</think>` token's start in
    `prompt_text + response`, or -1."""
    full = prompt_text + response
    return full.find("</think>", len(prompt_text))


def char_to_token_after(offsets, char_idx: int) -> int | None:
    for tok_idx, (s, e) in enumerate(offsets):
        if s <= char_idx < e or s == char_idx:
            return tok_idx
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--steer_vec", required=True,
                        help=".npz with v_unit, h_mean_norm fields")
    parser.add_argument("--rollouts_jsonl", default=None,
                        help="optional: phase2a per_answer_correctness.jsonl to filter to")
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--n_rollouts_per_prompt", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.countdown import compute_score

    # Load steering vector
    d = np.load(args.steer_vec)
    v_unit = torch.from_numpy(d["v_unit"]).to("cuda", dtype=torch.bfloat16)
    h_mean_norm = float(d["h_mean_norm"])
    print(f"[steer] v_unit shape {v_unit.shape}, h_mean_norm={h_mean_norm:.2f}", flush=True)

    # Load model + tokenizer
    print(f"[steer] loading model {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    # Random direction control (fixed per run for reproducibility)
    rng = np.random.RandomState(args.seed)
    r = rng.randn(896).astype(np.float32)
    r /= np.linalg.norm(r)
    v_rand = torch.from_numpy(r).to("cuda", dtype=torch.bfloat16)

    # Load rollouts
    rows = [json.loads(l) for l in open(args.eval_json) if l.strip()]
    if args.rollouts_jsonl:
        # filter to (prompt_idx, resp_idx) pairs in the rollouts JSONL
        valid = set()
        for line in open(args.rollouts_jsonl):
            r = json.loads(line)
            valid.add((int(r["prompt_idx"]), int(r["resp_idx"])))
    else:
        valid = None

    # Build list of (prompt_idx, resp_idx) to steer.
    work: list[tuple[int, int]] = []
    for p_idx, row in enumerate(rows):
        if p_idx >= args.n_prompts:
            break
        for r_idx in range(min(args.n_rollouts_per_prompt, len(row["response"]))):
            if valid is not None and (p_idx, r_idx) not in valid:
                continue
            work.append((p_idx, r_idx))
    print(f"[steer] will run {len(work)} (prompt, resp) prefixes x {len(args.alphas)} alphas x 3 directions (zero/probe/rand)", flush=True)

    # We use a forward hook on the decoder layer to inject the patch into the
    # post-layer residual stream at the </think> token, on the FIRST forward
    # pass only (prefix processing). Subsequent forward passes (autoregressive)
    # have seq_len == 1 and shouldn't be patched (they're generating new tokens).
    state = {"patch_token_pos": None, "patch_vec": None, "applied": False}

    def hook(module, inputs, outputs):
        # `outputs` may be a Tensor or a tuple (depending on transformers version)
        if state["patch_vec"] is None:
            return outputs
        is_tuple = isinstance(outputs, tuple)
        hs = outputs[0] if is_tuple else outputs
        if hs.dim() != 3 or hs.shape[1] == 1:
            # autoregressive step (single new token); don't patch
            return outputs
        pos = state["patch_token_pos"]
        if pos is None or pos >= hs.shape[1]:
            return outputs
        # Add patch vector at the specified position
        patch = state["patch_vec"].to(hs.dtype)
        new_hs = hs.clone()
        new_hs[0, pos] = new_hs[0, pos] + patch
        state["applied"] = True
        if is_tuple:
            return (new_hs,) + outputs[1:]
        return new_hs

    target_layer = model.model.layers[args.layer]
    handle = target_layer.register_forward_hook(hook)

    out_f = open(args.output_jsonl, "w")
    n_done = 0
    for p_idx, r_idx in work:
        row = rows[p_idx]
        prompt_text = row["prompt"]
        response = row["response"][r_idx]
        target = int(row["target"])
        nums = list(row["nums"])
        ground_truth = {"target": target, "numbers": nums}
        original_score = float(row["scores"][r_idx])

        # Find </think> in response
        think_char = find_think_close_char(prompt_text, response)
        if think_char < 0:
            continue
        # Prefix = prompt + response[:start_of_</think>] + "</think>"
        prefix = prompt_text + response[:think_char - len(prompt_text)] + "</think>"
        # Tokenize
        enc = tokenizer(prefix, return_offsets_mapping=True, truncation=True,
                        max_length=2048, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
        # Position of </think> token = the LAST token of the prefix
        think_tok_pos = input_ids.shape[1] - 1

        for alpha in args.alphas:
            for direction_name in ("zero", "probe", "rand"):
                if direction_name == "zero":
                    if alpha != 0.0:  # only do zero once
                        continue
                    state["patch_vec"] = None
                elif direction_name == "probe":
                    if alpha == 0.0:
                        continue
                    state["patch_vec"] = (alpha * h_mean_norm) * v_unit
                else:  # rand
                    if alpha == 0.0:
                        continue
                    state["patch_vec"] = (alpha * h_mean_norm) * v_rand
                state["patch_token_pos"] = think_tok_pos
                state["applied"] = False

                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=False)
                new_score = compute_score(new_text, ground_truth)

                out_f.write(json.dumps({
                    "prompt_idx": p_idx, "resp_idx": r_idx,
                    "alpha": float(alpha), "direction": direction_name,
                    "patch_applied": bool(state["applied"]),
                    "new_score": float(new_score),
                    "original_score": original_score,
                    "target": target, "nums": nums,
                    "new_completion": new_text[:1500],  # truncate to keep file size sane
                }) + "\n")
                out_f.flush()
        n_done += 1
        print(f"[steer] processed {n_done}/{len(work)} prefixes", flush=True)

    out_f.close()
    handle.remove()
    print(f"[steer] wrote {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
