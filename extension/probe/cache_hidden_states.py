"""Cache Qwen2.5-0.5B hidden states for the probe pipeline.

Per rollout in an eval JSON, runs a single forward pass with
`output_hidden_states=True` and extracts hidden states at three position kinds:

    pre_answer  - the `</think>` token (the standard probe-AUROC position;
                  one vector per rollout)
    assertion   - each occurrence of a confidence-asserting keyword inside
                  the <think> body (variable per rollout, capped)
    neutral     - matched-count token positions sampled from the <think>
                  body, excluding assertion positions (control for H3)

Output: one .npz per (checkpoint, layer, position) tuple containing
    X : (N, hidden_dim) float32
    y : (N,) int32          (1 = correct rollout, 0 = wrong)
plus a sidecar .meta.json with per-row provenance.

Cost: ~30 min on one H100 for 50 prompts × 16 rollouts × 3 layers, per
checkpoint. Bf16 forward; only the requested positions are kept, so disk
footprint is modest (~30 MB per .npz).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

# Make namespace imports work whether launched as module or script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


CONFIDENT_KEYWORDS: tuple[str, ...] = (
    "Perfect",
    "this works",
    "got it",
    "the answer is",
    "verified",
)

THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Text-side helpers.
# ---------------------------------------------------------------------------


def find_assertion_char_positions(
    think_body: str, keywords: Sequence[str]
) -> list[tuple[int, str]]:
    """Return char offsets of every match of any keyword, case-insensitive, sorted."""
    out: list[tuple[int, str]] = []
    lower = think_body.lower()
    for kw in keywords:
        target = kw.lower()
        start = 0
        while True:
            idx = lower.find(target, start)
            if idx < 0:
                break
            out.append((idx, kw))
            start = idx + 1
    out.sort(key=lambda x: x[0])
    return out


def char_to_token_index(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    """Map a char offset to the token whose (start, end) range contains it."""
    for tok_idx, (start, end) in enumerate(offsets):
        if start <= char_idx < end:
            return tok_idx
    return None


def first_token_at_or_after(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    for tok_idx, (start, _end) in enumerate(offsets):
        if start >= char_idx:
            return tok_idx
    return None


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="HF repo id or local path to the checkpoint to probe.")
    parser.add_argument("--eval_json", required=True,
                        help="Path to eval JSON produced by countdown_eval.py "
                             "(must have prompt/response/scores fields).")
    parser.add_argument("--checkpoint_name", required=True,
                        help="Short tag for output filenames (e.g. C_SFT, C_outcome).")
    parser.add_argument("--output_dir", default="/vol/probe_cache",
                        help="Where to write .npz files. On Modal use /vol/...")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20],
                        help="Transformer layer indices to extract.")
    parser.add_argument("--max_responses_per_prompt", type=int, default=16)
    parser.add_argument("--max_assertions_per_response", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--neutral_seed", type=int, default=0)
    args = parser.parse_args()

    # Heavy imports deferred so `--help` works without torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[cache] Loading tokenizer + model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required (need offset_mapping).")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    print(f"[cache] Loading eval JSON from {args.eval_json}")
    rows = [
        json.loads(line)
        for line in open(args.eval_json).read().strip().splitlines()
        if line.strip()
    ]
    print(f"[cache]   {len(rows)} prompts; up to {args.max_responses_per_prompt} responses each")

    # Storage per (layer, kind) -> {X: [], y: [], meta: []}.
    cache: dict[tuple[int, str], dict[str, list]] = {
        (layer, kind): {"X": [], "y": [], "meta": []}
        for layer in args.layers
        for kind in ("pre_answer", "assertion", "neutral")
    }

    rng = np.random.RandomState(args.neutral_seed)

    n_rollouts = 0
    n_rollouts_with_think_close = 0
    for prompt_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        responses = row["response"][: args.max_responses_per_prompt]
        scores = row["scores"][: args.max_responses_per_prompt]
        prompt_end_char = len(prompt_text)

        for resp_idx, (response, score) in enumerate(zip(responses, scores)):
            n_rollouts += 1
            label = int(score == 1.0)
            full_text = prompt_text + response

            enc = tokenizer(
                full_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to("cuda")
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            T = input_ids.shape[1]

            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True)
            # out.hidden_states is a tuple (n_layers + 1) of (1, T, hidden_dim) bf16.
            hidden_states = out.hidden_states

            # --- pre_answer: hidden state at the response's </think> token ---
            # The chat-templated prompt itself contains "<think> </think>" in
            # its instructions, so we have to search *after* the prompt to
            # find the model's actual closing tag.
            think_close_char = full_text.find(THINK_CLOSE, prompt_end_char)
            think_close_tok: int | None = None
            if think_close_char >= 0:
                think_close_tok = char_to_token_index(offsets, think_close_char)
            if think_close_tok is not None:
                n_rollouts_with_think_close += 1
                for layer in args.layers:
                    vec = hidden_states[layer][0, think_close_tok].float().cpu().numpy()
                    cache[(layer, "pre_answer")]["X"].append(vec)
                    cache[(layer, "pre_answer")]["y"].append(label)
                    cache[(layer, "pre_answer")]["meta"].append(
                        {"prompt_idx": prompt_idx, "resp_idx": resp_idx,
                         "tok_idx": think_close_tok}
                    )

            # --- assertion: per-keyword positions inside the response's
            # first <think> body (between the prompt and the first </think>
            # in the response). Restricting to the response prevents matches
            # of instruction-side text and keeps char offsets aligned with
            # `offsets` (which are full_text-indexed).
            think_body_end_char = think_close_char if think_close_char >= 0 else len(full_text)
            response_think_body = full_text[prompt_end_char:think_body_end_char]
            raw_hits = find_assertion_char_positions(response_think_body, CONFIDENT_KEYWORDS)
            raw_hits = raw_hits[: args.max_assertions_per_response]
            assertion_hits = [(c + prompt_end_char, kw) for c, kw in raw_hits]

            assertion_toks: list[int] = []
            for char_idx, kw in assertion_hits:
                tok_idx = char_to_token_index(offsets, char_idx)
                if tok_idx is None:
                    continue
                assertion_toks.append(tok_idx)
                for layer in args.layers:
                    vec = hidden_states[layer][0, tok_idx].float().cpu().numpy()
                    cache[(layer, "assertion")]["X"].append(vec)
                    cache[(layer, "assertion")]["y"].append(label)
                    cache[(layer, "assertion")]["meta"].append(
                        {"prompt_idx": prompt_idx, "resp_idx": resp_idx,
                         "tok_idx": tok_idx, "keyword": kw}
                    )

            # --- neutral: matched-count sample from response tokens ---------
            if think_close_tok is not None and assertion_toks:
                prompt_end_tok = first_token_at_or_after(offsets, prompt_end_char)
                if prompt_end_tok is None:
                    prompt_end_tok = 0
                pool = [
                    i
                    for i in range(prompt_end_tok + 1, think_close_tok)
                    if i not in assertion_toks
                ]
                n_take = min(len(assertion_toks), len(pool))
                if n_take > 0:
                    chosen = rng.choice(pool, size=n_take, replace=False)
                    for tok_idx in chosen:
                        for layer in args.layers:
                            vec = hidden_states[layer][0, int(tok_idx)].float().cpu().numpy()
                            cache[(layer, "neutral")]["X"].append(vec)
                            cache[(layer, "neutral")]["y"].append(label)
                            cache[(layer, "neutral")]["meta"].append(
                                {"prompt_idx": prompt_idx, "resp_idx": resp_idx,
                                 "tok_idx": int(tok_idx)}
                            )

        if (prompt_idx + 1) % 10 == 0:
            print(f"[cache]   processed {prompt_idx + 1}/{len(rows)} prompts")

    print(f"[cache] Done. {n_rollouts} rollouts; "
          f"{n_rollouts_with_think_close} had a locatable </think> token.")

    # Write outputs.
    print(f"[cache] Writing to {args.output_dir}/")
    for (layer, kind), data in cache.items():
        if not data["X"]:
            print(f"[cache]   ({layer}, {kind}): empty, skipping")
            continue
        X = np.stack(data["X"], axis=0).astype(np.float32)
        y = np.array(data["y"], dtype=np.int32)
        base = f"{args.checkpoint_name}_l{layer}_{kind}"
        np.savez_compressed(os.path.join(args.output_dir, f"{base}.npz"), X=X, y=y)
        with open(os.path.join(args.output_dir, f"{base}.meta.json"), "w") as f:
            json.dump(data["meta"], f)
        print(f"[cache]   ({layer}, {kind}): X={X.shape} y={y.shape} "
              f"-> {base}.npz")


if __name__ == "__main__":
    main()
