"""Elicit verbalized confidence from a checkpoint on each of its eval rollouts.

For each (prompt, response, score) in the eval JSON, prompts the same model
that produced the response with a confidence-elicitation template and parses
an integer in [0, 100]. Writes one JSONL row per rollout to --output_json.

Output schema:
  {prompt_idx, resp_idx, target, nums, answer, score, verbalized_confidence}

The verbalized_confidence field is None when parsing failed.

Designed to be launched on Modal (needs GPU for vLLM batched inference).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Prompt template.
# ---------------------------------------------------------------------------

CONFIDENCE_PROMPT = (
    "You previously solved a Countdown arithmetic problem. Below is the "
    "problem and your candidate equation. Rate your confidence on a scale "
    "from 0 to 100 that the candidate equation is correct (i.e. uses each "
    "provided number exactly once and evaluates to the target).\n\n"
    "Problem: Using the numbers {nums}, create an equation that equals {target}.\n"
    "Candidate equation: {answer}\n\n"
    "Reply with ONLY an integer in [0, 100]. No words, no punctuation, "
    "nothing else.\nConfidence (0-100):"
)


_INT_RE = re.compile(r"\b(\d{1,3})\b")


def parse_confidence(text: str) -> int | None:
    m = _INT_RE.search(text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except ValueError:
        return None
    if 0 <= v <= 100:
        return v
    return None


def extract_last_answer(response: str) -> str | None:
    # Mirrors evaluation/countdown.py extract_solution semantics but only
    # returns the last <answer>...</answer> body's raw text.
    m = list(re.finditer(r"<answer>(.*?)</answer>", response or "", re.DOTALL))
    if not m:
        return None
    return m[-1].group(1).strip().rstrip(".")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_json", required=True,
                        help="Eval JSON produced by countdown_eval.py.")
    parser.add_argument("--output_json", required=True,
                        help="JSONL output with one row per (prompt, response) pair.")
    parser.add_argument("--max_responses_per_prompt", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=8,
                        help="Generation budget for the integer reply.")
    args = parser.parse_args()

    # Lazy import vLLM so --help works without it.
    from vllm import LLM, SamplingParams

    print(f"[elicit] loading model {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    print(f"[elicit] loading eval JSON {args.eval_json}")
    rows = [
        json.loads(line)
        for line in open(args.eval_json).read().strip().splitlines()
        if line.strip()
    ]
    print(f"[elicit]   {len(rows)} prompts")

    # Build all prompts for batched inference.
    items: list[dict] = []   # bookkeeping per call
    prompts: list[str] = []
    for prompt_idx, row in enumerate(rows):
        target = int(row["target"])
        nums = list(row["nums"])
        responses = row["response"][: args.max_responses_per_prompt]
        scores = row["scores"][: args.max_responses_per_prompt]
        for resp_idx, (resp, score) in enumerate(zip(responses, scores)):
            answer = extract_last_answer(resp) or "(no answer parsed)"
            prompt = CONFIDENCE_PROMPT.format(
                nums=str(list(nums)), target=target, answer=answer,
            )
            prompts.append(prompt)
            items.append({
                "prompt_idx": prompt_idx,
                "resp_idx": resp_idx,
                "target": target,
                "nums": nums,
                "answer": answer,
                "score": float(score),
            })
    print(f"[elicit]   {len(prompts)} confidence calls to run")

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)

    n_parsed = 0
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        for item, out in zip(items, outputs):
            text = out.outputs[0].text if out.outputs else ""
            conf = parse_confidence(text)
            if conf is not None:
                n_parsed += 1
            item_out = dict(item)
            item_out["raw_completion"] = text
            item_out["verbalized_confidence"] = conf
            f.write(json.dumps(item_out) + "\n")
    print(f"[elicit] done. {n_parsed}/{len(items)} confidences parsed.")
    print(f"[elicit] wrote {args.output_json}")


if __name__ == "__main__":
    main()
