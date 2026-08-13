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
output for the `</think>` token position, during the prefill pass only.
The modified residual then propagates through layers L+1..N for that
position, so their K/V entries for `</think>` are built from the patched
value and every later generated token attending to `</think>` sees it.
(Layer L's own K/V for that position is computed from layer L's *input*
and is therefore unchanged -- the intervention is strictly downstream of L,
which is the right scope for a direction read off layer L's output.)

The injection lands on the token containing the FIRST character of
`</think>`, matching the position cache_hidden_states.py used to fit the
probe direction. See --steer_position.

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
    parser.add_argument("--layer", type=int, default=16,
                        help="The hidden_states index the probe was fit on. See "
                             "--layer_convention for which block that hooks.")
    parser.add_argument(
        "--layer_convention", choices=("hidden_state", "legacy_block"),
        default="hidden_state",
        help="How --layer maps to a decoder block. 'hidden_state' (default, "
             "correct) treats --layer L as the hidden_states index the probe "
             "reads and hooks model.layers[L-1], since hidden_states[0] is the "
             "embedding output. 'legacy_block' hooks model.layers[L], which "
             "writes to hidden_states[L+1] -- one block downstream of the read "
             "site. That was the original behaviour and is kept only to "
             "reproduce the shipped JSONLs.")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steer_position", choices=("probe_read", "last_token"), default="probe_read",
        help="Which token of `</think>` to inject at. 'probe_read' (default, correct) "
             "is the token containing the FIRST character of '</think>' -- the exact "
             "position cache_hidden_states.py read to fit the probe direction. "
             "'last_token' is the final token of the prefix (the trailing '>'), which "
             "is 2-3 positions LATER; that was the original behaviour and is kept only "
             "to reproduce the earlier runs. Steering somewhere the probe does not read "
             "does not test whether the probe direction is causal.")
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
    # Random direction control (fixed per run for reproducibility). Dimension is
    # taken from the steering vector rather than hardcoded to 896, so this also
    # works at 1.5B (hidden_size=1536) instead of failing on a broadcast error.
    rng = np.random.RandomState(args.seed)
    d_model = int(v_unit.shape[0])
    if d_model != model.config.hidden_size:
        raise ValueError(f"steer_vec dim {d_model} != model hidden_size "
                         f"{model.config.hidden_size}; wrong vector for this model?")
    r = rng.randn(d_model).astype(np.float32)
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

    # WHICH BLOCK TO HOOK. `output_hidden_states=True` returns hidden_states[0] =
    # the embedding output, so hidden_states[L] is the output of layers[L-1].
    # cache_hidden_states.py fits the probe on hidden_states[16]; hooking
    # layers[16] therefore patches hidden_states[17] -- one full transformer
    # block DOWNSTREAM of the site the probe reads. That off-by-one is live in
    # every steering JSONL shipped before 2026-08-12, alongside the token-position
    # bug --steer_position now fixes, and it is why the published section 3 result
    # is reported as inconclusive rather than defended.
    #
    # `hidden_state` (default) patches the read site. `legacy_block` reproduces
    # the shipped runs bit-for-bit; use it only to regenerate old artifacts.
    if args.layer_convention == "hidden_state":
        hook_block = args.layer - 1
    else:
        hook_block = args.layer
    if not 0 <= hook_block < len(model.model.layers):
        raise SystemExit(
            f"[steer] layer {args.layer} under convention "
            f"{args.layer_convention!r} resolves to block {hook_block}, which is "
            f"outside 0..{len(model.model.layers) - 1}"
        )
    print(f"[steer] layer_convention={args.layer_convention}: probe reads "
          f"hidden_states[{args.layer}]; hooking model.layers[{hook_block}]",
          flush=True)
    target_layer = model.model.layers[hook_block]
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

        # Where to inject. The probe direction was fitted on the hidden state at
        # the token CONTAINING THE FIRST CHARACTER of "</think>"
        # (cache_hidden_states.py -> char_to_token_index). "</think>" is 2-3
        # tokens wide for this tokenizer, so the last token of the prefix (the
        # trailing '>') is NOT the position the probe reads. Injecting there
        # tests a different position than the one the null result is about.
        if args.steer_position == "last_token":
            think_tok_pos = input_ids.shape[1] - 1
        else:
            close_char_in_prefix = len(prefix) - len("</think>")
            think_tok_pos = char_to_token_after(offsets, close_char_in_prefix)
            if think_tok_pos is None:
                print(f"[steer] WARNING: could not locate </think> token for "
                      f"({p_idx},{r_idx}); skipping", flush=True)
                continue
        if n_done == 0:
            span = tokenizer.decode(input_ids[0, think_tok_pos:].tolist())
            print(f"[steer] steer_position={args.steer_position}: injecting at token "
                  f"{think_tok_pos}/{input_ids.shape[1]-1}; text from there = {span!r}",
                  flush=True)

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
                    "steer_position": args.steer_position,
                    "layer_convention": args.layer_convention,
                    "probe_hidden_state_layer": int(args.layer),
                    "hook_block": int(hook_block),
                    "steer_tok_pos": int(think_tok_pos),
                    "n_prefix_tokens": int(input_ids.shape[1]),
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
