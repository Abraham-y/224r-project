"""vLLM rollout sampler that reads prompts from a local JSONL.

Adapted from `evaluation/countdown_eval.py` (which only supports HuggingFace
dataset names). The expanded evaluation set was generated procedurally and
lives at `extension/data/countdown_eval_500.jsonl`, which is bundled into
the Modal image at `/root/default_proj/extension/data/countdown_eval_500.jsonl`.

For each (prompt, target, nums) row in the input JSONL, samples
`--num_responses` rollouts via vLLM and scores them with the same verifier
used during training (`evaluation.countdown.compute_score`).

Output: one JSON file in the same row schema as eval.json:
    {prompt, target, nums, ground_truth, response: [str x N], scores: [float x N]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# So `from evaluation.countdown import compute_score` works in Modal.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.countdown import compute_score


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def load_checkpoint(
    model_path: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
    max_num_seqs: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=max_num_seqs,
    )
    return tokenizer, llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True,
                        help="Path to a JSONL where each row has prompt, target, nums.")
    parser.add_argument("--output_json", required=True,
                        help="Where to write the per-prompt rollouts + scores.")
    parser.add_argument("--num_responses", type=int, default=16)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Optional cap on number of prompts (uses the first --max_prompts rows).")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--enable_chunked_prefill", type=bool, default=True)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument(
        "--extra_stop_token_ids",
        type=str,
        default="",
        help="Comma-separated extra token IDs to treat as stops. "
             "E.g. '151645' to also stop on <|im_end|>. Empty = default (eos_token_id only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[eval] loading prompts from {args.input_jsonl}")
    with open(args.input_jsonl) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_prompts is not None:
        rows = rows[: args.max_prompts]
    print(f"[eval]   {len(rows)} prompts")

    print(f"[eval] loading model {args.model_path}")
    tokenizer, llm = load_checkpoint(
        args.model_path,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.max_num_batched_tokens,
        args.enable_chunked_prefill,
        args.max_num_seqs,
    )

    extra_stops = [int(x) for x in args.extra_stop_token_ids.split(",") if x.strip()]
    stop_ids = [tokenizer.eos_token_id] + extra_stops
    print(f"[eval] sampling will stop on token ids: {stop_ids} "
          f"(decoded: {[tokenizer.decode([i]) for i in stop_ids]!r})")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        n=args.num_responses,
        stop_token_ids=stop_ids,
    )

    prompts = [r["prompt"] for r in rows]
    print(f"[eval] generating {args.num_responses} responses per prompt "
          f"({len(prompts) * args.num_responses} total) ...")
    outputs = llm.generate(prompts, sampling_params)

    responses_per_prompt: list[list[str]] = []
    scores_per_prompt: list[list[float]] = []
    for i, output in enumerate(outputs):
        row = rows[i]
        ground_truth = {"target": row["target"], "numbers": row["nums"]}
        curr_resps: list[str] = []
        curr_scores: list[float] = []
        for o in output.outputs:
            text = o.text
            score = compute_score(text, ground_truth)
            curr_resps.append(text)
            curr_scores.append(float(score))
        responses_per_prompt.append(curr_resps)
        scores_per_prompt.append(curr_scores)

    out_rows: list[dict] = []
    for row, resps, scs in zip(rows, responses_per_prompt, scores_per_prompt):
        out = {
            "prompt": row["prompt"],
            "target": int(row["target"]),
            "nums": list(row["nums"]),
            "ground_truth": row.get("ground_truth", {"numbers": list(row["nums"]), "target": int(row["target"])}),
            "response": resps,
            "scores": scs,
        }
        out_rows.append(out)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # Quick summary.
    pass_at_1 = sum(s[0] == 1.0 for s in scores_per_prompt) / max(1, len(scores_per_prompt))
    pass_at_k = sum(any(x == 1.0 for x in s) for s in scores_per_prompt) / max(1, len(scores_per_prompt))
    print(f"[eval] wrote {args.output_json}")
    print(f"[eval] pass@1 = {pass_at_1:.3f}, pass@{args.num_responses} = {pass_at_k:.3f}")


if __name__ == "__main__":
    main()
