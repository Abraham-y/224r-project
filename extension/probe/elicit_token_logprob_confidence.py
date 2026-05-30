"""Token-logprob-based verbalized confidence elicitation.

For each rollout in an eval JSON, build a prompt = (chat-templated problem +
model's response) + a verdict-eliciting follow-up turn, then run a SINGLE
forward pass that returns top-K next-token logprobs. Extract
`logprob(yes_variants)` and `logprob(no_variants)` and report:
  - logit_gap = logprob(best yes token) - logprob(best no token)
  - p_yes_normalized = exp(yes) / (exp(yes) + exp(no))

This is the literature-standard verbalized-confidence elicitation when the
base model is not chat-tuned (RLCR / Yuan et al. variants). It works
because we never need the model to *generate* a fluent answer -- we just
need its first-token distribution.

Output: JSONL, one row per (prompt, response) pair with the fields:
  prompt_idx, resp_idx, target, nums, score,
  logprob_yes, logprob_no, logit_gap, p_yes_normalized,
  raw_first_token (for sanity).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


# After the model's response we append a chat-template user turn that asks
# for a yes/no verdict, then open the assistant turn so the next generated
# token is the verdict itself. The eval prompt template uses Qwen2.5's
# <|im_start|>/<|im_end|> markers; we close any dangling assistant turn and
# add a new user turn.
CONFIDENCE_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Was the candidate equation above correct? Reply with exactly one word: "
    "yes or no.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def discover_yes_no_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """Map common yes/no surface forms to single-token IDs (where they tokenize as 1 token)."""
    yes_ids: set[int] = set()
    no_ids: set[int] = set()
    yes_words = ("yes", "Yes", "YES", " yes", " Yes", " YES")
    no_words = ("no", "No", "NO", " no", " No", " NO")
    for w in yes_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for w in no_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])
    return yes_ids, no_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_responses_per_prompt", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--logprobs_top_k", type=int, default=20,
                        help="vLLM top-K logprobs to request per generated token. "
                             "vLLM caps this at 20 by default; the yes/no tokens are "
                             "near-certainly within the top 20 for a verdict prompt.")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams  # heavy import; defer

    print(f"[logprob] loading model {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    yes_ids, no_ids = discover_yes_no_token_ids(tokenizer)
    print(f"[logprob] yes token ids: {sorted(yes_ids)}")
    print(f"[logprob] no  token ids: {sorted(no_ids)}")
    if not yes_ids or not no_ids:
        raise SystemExit("Could not find single-token yes/no IDs in this tokenizer.")

    print(f"[logprob] loading eval JSON {args.eval_json}")
    rows = [
        json.loads(line)
        for line in open(args.eval_json).read().strip().splitlines()
        if line.strip()
    ]
    print(f"[logprob]   {len(rows)} prompts")

    items: list[dict] = []
    prompts: list[str] = []
    for prompt_idx, row in enumerate(rows):
        responses = row["response"][: args.max_responses_per_prompt]
        scores = row["scores"][: args.max_responses_per_prompt]
        for resp_idx, (resp, score) in enumerate(zip(responses, scores)):
            # Build: (chat-templated prompt + response so far) + verdict suffix.
            full = row["prompt"] + resp + CONFIDENCE_SUFFIX
            prompts.append(full)
            items.append({
                "prompt_idx": prompt_idx,
                "resp_idx": resp_idx,
                "target": int(row["target"]),
                "nums": list(row["nums"]),
                "score": float(score),
            })
    print(f"[logprob]   {len(prompts)} verdict queries")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=args.logprobs_top_k,
    )
    outputs = llm.generate(prompts, sampling)

    n_with_both = 0
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        for item, out in zip(items, outputs):
            first_tok = out.outputs[0]
            raw_first = first_tok.text or ""
            logprobs_at_0 = first_tok.logprobs[0] if first_tok.logprobs else {}

            best_yes = -math.inf
            best_no = -math.inf
            for tok_id, lp in logprobs_at_0.items():
                # vLLM's Logprob object has .logprob; coerce to float.
                value = float(getattr(lp, "logprob", lp))
                if tok_id in yes_ids and value > best_yes:
                    best_yes = value
                if tok_id in no_ids and value > best_no:
                    best_no = value

            item_out = dict(item)
            item_out["raw_first_token"] = raw_first
            item_out["logprob_yes"] = best_yes if best_yes != -math.inf else None
            item_out["logprob_no"] = best_no if best_no != -math.inf else None
            if best_yes != -math.inf and best_no != -math.inf:
                item_out["logit_gap"] = best_yes - best_no
                m = max(best_yes, best_no)
                item_out["p_yes_normalized"] = (
                    math.exp(best_yes - m) / (math.exp(best_yes - m) + math.exp(best_no - m))
                )
                n_with_both += 1
            else:
                item_out["logit_gap"] = None
                item_out["p_yes_normalized"] = None
            f.write(json.dumps(item_out) + "\n")

    print(f"[logprob] {n_with_both}/{len(items)} rows have both yes and no in top-{args.logprobs_top_k}")
    print(f"[logprob] wrote {args.output_json}")


if __name__ == "__main__":
    main()
