"""Train + pickle the `<answer>`-opening L16 correctness probe used by
`probe_behavioral_correlation.py`.

Why this script exists. The repo's earlier saved probe directions live at
*other* positions: `extension/probe/save_probe_vector.py` saves a direction
at the `</think>` (trace-final / pre_answer) position, and the "assertion"
probe is at confidence-keyword tokens. The Phase 2A work (§2.4 / §8 of
writeup.md) trained a position-appropriate probe at the **`<answer>` opening
token** (per-block correctness labels, position-appropriate diagonal AUROC =
**0.920**). This script deliberately reproduces THAT position
(`--answer_token_pos open`, default) so the experiment's probe lives on the
same linear position as the writeup's position-appropriate probe. Only the
*labels* differ -- the writeup probe uses per-block correctness; this probe
uses rollout-final correctness, which is the right label for a per-problem
behavioral-correlation question (since accuracy = pass@1 = "did the FINAL
scored <answer> match the target?").

Sanity bar. Held-out (by problem) AUROC should land near §2.4's 0.920.
If it diverges by more than ~0.05, something is off (likely the contaminated-
94 training pool is too thin) and we should investigate before trusting
per-problem AUROCs.

No new data. This script generates NOTHING -- no synthetic problems, no new
rollouts. It reuses the **existing** C_SFT rollouts already on the volume
(`/vol/evaluation/eval_results/eval_c_sft_n500.json`, produced by the n=500
expansion) and only runs forward passes to extract hidden-state features (the
same operation as `cache_answer_positions.py`).

No train/eval leakage. The experiment scores the clean-406 problems. We train
the probe ONLY on the 94 *contaminated* problems (those dropped from clean-406
because they are in C_outcome's RLOO train set). Those 94 are disjoint from the
clean-406 eval set by construction, so the probe never sees a problem the
experiment scores. (The "contamination" label is about C_outcome's RL training
and is irrelevant to a C_SFT decoder; what matters here is only disjointness.)

Consistency. The `<answer>`-opening token is located with the exact same
`find_assertion_token` used by the experiment (imported, not reimplemented).
The probe is pickled as a Pipeline(StandardScaler, LogisticRegression) so the
scaling travels inside the pickle and `load_probe_scorer` applies it.

GPU phase (forward passes only); intended for Modal:

    modal run modal_train.py train_answer_probe -- \
        --model asingh15/qwen-sft-countdown-defaultproj \
        --rollouts_json /vol/evaluation/eval_results/eval_c_sft_n500.json \
        --out /vol/outputs/probe_behavioral/probe.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from extension.probe.probe_behavioral_correlation import find_assertion_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="asingh15/qwen-sft-countdown-defaultproj",
                        help="Checkpoint that PRODUCED the rollouts (C_SFT). Its hidden "
                             "states are extracted, so it must match --rollouts_json.")
    parser.add_argument("--rollouts_json",
                        default="/vol/evaluation/eval_results/eval_c_sft_n500.json",
                        help="Existing C_SFT rollouts (prompt/target/nums/response/scores "
                             "per row, row i == problem i in countdown_eval_500.jsonl).")
    parser.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    parser.add_argument("--train_split", choices=("contaminated", "all"), default="contaminated",
                        help="'contaminated': train on the 94 problems disjoint from the "
                             "clean-406 eval set (leakage-free, default). 'all': every "
                             "problem (WARNING: overlaps the eval set).")
    parser.add_argument("--max_responses_per_prompt", type=int, default=16)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--answer_occurrence", choices=("first", "last"), default="last",
                        help="'last' matches the verifier (which scores the final <answer>) "
                             "and the writeup's Phase 2A convention.")
    parser.add_argument("--answer_token_pos", choices=("after", "open"), default="open",
                        help="'open' (default) places the probe at the <answer> opening "
                             "token -- the SAME position as the §2.4 position-appropriate "
                             "probe (held-out AUROC 0.920). 'after' uses the token "
                             "immediately following the tag, ~one token later in the "
                             "residual stream.")
    parser.add_argument("--C", type=float, default=0.1, help="LogisticRegression inverse reg.")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--out", default="/vol/outputs/probe_behavioral/probe.pkl")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from evaluation.countdown import compute_score

    rows = [json.loads(l) for l in open(args.rollouts_json) if l.strip()]
    print(f"[train-probe] loaded {len(rows)} problems from {args.rollouts_json}", flush=True)

    # Select the leakage-free training problem indices.
    if args.train_split == "contaminated":
        contam = set(json.load(open(args.contam_json))["contaminated"])
        train_idx = [i for i in range(len(rows)) if i in contam]
        print(f"[train-probe] training on {len(train_idx)} contaminated problems "
              f"(disjoint from clean-406 eval)", flush=True)
    else:
        train_idx = list(range(len(rows)))
        print(f"[train-probe] WARNING: training on ALL {len(train_idx)} problems -- this "
              f"OVERLAPS the clean-406 eval set (leakage).", flush=True)

    print(f"[train-probe] loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required (need offset_mapping).")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    X_list, y_list, g_list = [], [], []
    n_no_answer = 0
    for n_done, p_idx in enumerate(train_idx):
        row = rows[p_idx]
        prompt_text = row["prompt"]
        ground_truth = {"target": int(row["target"]), "numbers": list(row["nums"])}
        responses = row["response"][: args.max_responses_per_prompt]
        scores = row["scores"][: args.max_responses_per_prompt]
        prompt_end_char = len(prompt_text)
        for response, score in zip(responses, scores):
            label = int(float(score) == 1.0)
            full_text = prompt_text + response
            enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True,
                            max_length=args.max_seq_len, return_tensors="pt")
            input_ids = enc["input_ids"].to("cuda")
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            tok_idx = find_assertion_token(offsets, response, prompt_end_char,
                                           args.answer_occurrence, args.answer_token_pos)
            if tok_idx is None or tok_idx >= input_ids.shape[1]:
                n_no_answer += 1
                continue
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True)
            vec = out.hidden_states[args.layer][0, tok_idx].float().cpu().numpy()
            X_list.append(vec); y_list.append(label); g_list.append(p_idx)
        if (n_done + 1) % 25 == 0:
            print(f"[train-probe] {n_done + 1}/{len(train_idx)} problems "
                  f"({len(y_list)} usable rollouts)", flush=True)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(g_list, dtype=np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"[train-probe] {len(y)} usable rollouts ({n_no_answer} had no <answer>); "
          f"pos={n_pos} neg={n_neg}", flush=True)
    if n_pos < 10 or n_neg < 10:
        raise RuntimeError(f"Too few of one class (pos={n_pos}, neg={n_neg}). Consider a "
                           f"larger --rollouts_json or --train_split all (with leakage).")

    # Balanced subsample (mirrors save_probe_vector.py).
    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, nb, replace=False),
                          rng.choice(neg, nb, replace=False)])
    Xs, ys, gs = X[idx], y[idx], groups[idx]

    def make_pipe():
        return Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(C=args.C, max_iter=2000))])

    # Held-out-by-problem sanity AUROC.
    n_splits = min(5, len(np.unique(gs)))
    preds = np.full(len(ys), np.nan)
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits).split(Xs, ys, gs):
            preds[te] = make_pipe().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
    m = ~np.isnan(preds)
    holdout_auroc = (float(roc_auc_score(ys[m], preds[m]))
                     if m.any() and len(set(ys[m].tolist())) == 2 else float("nan"))
    print(f"[train-probe] held-out (by problem) AUROC = {holdout_auroc:.3f} on "
          f"{nb}+{nb} balanced post-<answer> L{args.layer} examples "
          f"({n_splits}-fold)", flush=True)

    pipe = make_pipe().fit(Xs, ys)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(pipe, f)
    meta = {
        "trained_on": args.model, "rollouts_json": args.rollouts_json,
        "train_split": args.train_split, "n_train_problems": len(train_idx),
        "layer": args.layer, "answer_occurrence": args.answer_occurrence,
        "answer_token_pos": args.answer_token_pos,
        "n_usable_rollouts": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
        "n_balanced_per_class": int(nb), "C": args.C,
        "holdout_auroc_by_problem": holdout_auroc,
        "pickle_kind": "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)",
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train-probe] wrote {args.out} (+ .meta.json)", flush=True)


if __name__ == "__main__":
    main()
