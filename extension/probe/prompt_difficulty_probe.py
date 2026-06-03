"""Prompt-only difficulty probe — a different tangent.

Every other probe in this project reads AFTER the model reasons (`</think>` /
`<answer>`) and predicts a *rollout's correctness*. This one reads the hidden
state at the **last prompt token** — the moment the model has read the problem
but generated nothing — and predicts the **problem's difficulty** (the per-
problem pass rate over K rollouts).

Question: does the model encode "how hard is this problem / will I solve it"
at *input time*, before any reasoning? If so it's a **zero-sample** difficulty
estimate (vs the existing probe-mean estimator, which needs K rollouts).

Controls:
  - surface-feature baseline (n_numbers, target, sum/max/min/range of nums):
    does the internal state beat trivially-available problem features?
  - shuffled-label baseline (should be ~chance / r~0).

Method: forward-pass each prompt in --rollouts_json once, take L<layer> hidden
at the final prompt token; label = pass rate from that json's scores; KFold(5):
  - Ridge -> predict pass rate (held-out Pearson/Spearman r)
  - LogReg -> solvable (pass_rate>0) AUROC
Reported for internal vs surface vs shuffle. Clean-406 filtered.

Runs on Modal (GPU forward passes + inline sklearn). Saves a JSON + scatter to
the volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def surface_features(nums, target):
    nums = [float(x) for x in nums]
    return [len(nums), float(target), sum(nums), max(nums), min(nums),
            max(nums) - min(nums), float(np.mean(nums)), float(np.std(nums))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="asingh15/qwen-sft-countdown-defaultproj")
    ap.add_argument("--rollouts_json", default="eval_c_sft_n500.json",
                    help="rollouts whose prompts we read and whose scores give pass rate")
    ap.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--clean_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--out_dir", default="/vol/outputs/prompt_difficulty")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr, spearmanr

    rows = [json.loads(l) for l in open(args.rollouts_json) if l.strip()]
    if args.clean_only and os.path.exists(args.contam_json):
        clean = set(json.load(open(args.contam_json))["clean"])
        rows = [r for i, r in enumerate(rows) if i in clean]
    print(f"[prompt-diff] {len(rows)} problems (clean_only={args.clean_only})", flush=True)

    print(f"[prompt-diff] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    X_int, X_surf, pass_rate = [], [], []
    for i, r in enumerate(rows):
        scores = r["scores"][:16]
        pr = float(np.mean([1.0 if s == 1.0 else 0.0 for s in scores]))
        enc = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, -1].float().cpu().numpy()  # last prompt token
        X_int.append(h)
        X_surf.append(surface_features(r["nums"], r["target"]))
        pass_rate.append(pr)
        if (i + 1) % 100 == 0:
            print(f"[prompt-diff]   {i+1}/{len(rows)} prompts", flush=True)

    X_int = np.asarray(X_int, np.float32)
    X_surf = np.asarray(X_surf, np.float32)
    y = np.asarray(pass_rate, np.float32)
    solvable = (y > 0).astype(int)
    print(f"[prompt-diff] pass_rate mean={y.mean():.3f} | solvable frac={solvable.mean():.3f}", flush=True)

    def evaluate(X, name, shuffle=False):
        yy = y.copy()
        ss = solvable.copy()
        if shuffle:
            rng = np.random.RandomState(0)
            perm = rng.permutation(len(yy)); yy = yy[perm]; ss = ss[perm]
        kf = KFold(5, shuffle=True, random_state=0)
        reg_pred = np.full(len(yy), np.nan)
        clf_pred = np.full(len(yy), np.nan)
        for tr, te in kf.split(X):
            reg = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=10.0))]).fit(X[tr], yy[tr])
            reg_pred[te] = reg.predict(X[te])
            if len(set(ss[tr].tolist())) == 2:
                clf = Pipeline([("s", StandardScaler()), ("l", LogisticRegression(C=0.1, max_iter=2000))]).fit(X[tr], ss[tr])
                clf_pred[te] = clf.predict_proba(X[te])[:, 1]
        r_p = float(pearsonr(yy, reg_pred)[0])
        r_s = float(spearmanr(yy, reg_pred)[0])
        au = (float(roc_auc_score(ss, clf_pred)) if len(set(ss.tolist())) == 2 and not np.isnan(clf_pred).all()
              else float("nan"))
        print(f"  {name:28} pass-rate Pearson r={r_p:+.3f} Spearman={r_s:+.3f} | solvable AUROC={au:.3f}", flush=True)
        return {"pearson_r": r_p, "spearman_r": r_s, "solvable_auroc": au}

    print("\n=== prompt-only difficulty prediction (held-out KFold5) ===", flush=True)
    res = {
        "internal_promptL%d" % args.layer: evaluate(X_int, f"internal (prompt L{args.layer})"),
        "surface_features": evaluate(X_surf, "surface features (control)"),
        "internal_shuffled": evaluate(X_int, "internal SHUFFLED (sanity)", shuffle=True),
    }
    res["_meta"] = {"model": args.model, "rollouts_json": args.rollouts_json,
                    "n_problems": len(rows), "layer": args.layer,
                    "pass_rate_mean": float(y.mean()), "solvable_frac": float(solvable.mean())}

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "prompt_difficulty.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[prompt-diff] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
