"""Per-problem probe-AUROC vs behavioral-accuracy-delta correlation.

Does the post-RL drop in the assertion-position probe's discriminative power
co-occur, *problem by problem*, with a change in the model's behavioral
accuracy? This script tests the hypothesis directly.

No new data. It reuses the **existing** rollouts already on the volume
(`eval_c_sft_n500.json` / `eval_c_outcome_n500.json`, 16 rollouts/problem, the
same rollouts behind the headline numbers) and only runs forward passes to
extract hidden-state features. Nothing is sampled or generated.

Pipeline (per the 5-step spec):

  Step 1 -- Per-problem probe AUROC under each checkpoint.
    For each problem, take its K existing rollouts. For each rollout, extract
    the layer-L hidden state at the assertion token (the token *immediately
    following* the `<answer>` tag; see --answer_occurrence / --answer_token_pos),
    run the linear probe on it to get a scalar score, and read the verifier's
    binary correctness label (stored score == 1.0). AUROC over the K
    (probe_score, label) pairs is the per-problem probe quality.
    => auroc_sft[i], auroc_rloo[i].

  Step 2 -- Accuracy delta per problem.
    accuracy_*[i] = fraction of K rollouts the verifier scored 1.0.
    accuracy_delta[i] = accuracy_rloo[i] - accuracy_sft[i].

  Step 3 -- Probe drop.
    probe_drop[i] = auroc_sft[i] - auroc_rloo[i].  (>0 means the probe got
    *worse* at reading correctness after RL on that problem.)

  Step 4 -- Spearman correlation between probe_drop and accuracy_delta across
    problems, plus a scatter plot with a regression line and quadrant labels.

  Step 5 -- Save probe_behavioral_correlation.png and a JSON with
    spearman_r, spearman_p, probe_drop, accuracy_delta, auroc_sft, auroc_rloo.

Edge case (AUROC undefined): a problem whose K rollouts are all-correct or
all-incorrect under a checkpoint has no defined AUROC. Such a problem keeps a
NaN AUROC (and NaN probe_drop) and is dropped from the Spearman correlation,
but its accuracy_delta is still computed and reported.

Contamination filter (default on): the correlation runs on the clean-406
problems (those NOT in C_outcome's RLOO train set), matching the paper's
headline set. Use --no-clean_only for all 500.

Alignment: both rollout JSONs are row-aligned to countdown_eval_500.jsonl
(row i == problem i), so per-problem comparison across checkpoints is by
original problem index.

Per-checkpoint per-rollout records (probe score + label) are cached to JSONL,
so re-plotting needs neither GPU nor the probe. Intended for Modal:

    modal run modal_train.py probe_behavioral -- \
        --probe /vol/outputs/probe_behavioral/probe.pkl \
        --output_dir /vol/outputs/probe_behavioral \
        --cache_dir /vol/outputs/probe_behavioral
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys

import numpy as np

# Make `from evaluation.countdown import compute_score` work whether this is
# launched as a module or a bare script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_ANSWER_OPEN_RE = re.compile(r"<answer>")


# ---------------------------------------------------------------------------
# Char -> token helpers (same logic as cache_answer_positions.py).
# ---------------------------------------------------------------------------


def char_to_token_index(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    """First token whose [start, end) interval contains (or starts at) char_idx."""
    for tok_idx, (s, e) in enumerate(offsets):
        if s <= char_idx < e or s == char_idx:
            return tok_idx
    return None


def first_token_at_or_after(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    for tok_idx, (s, _e) in enumerate(offsets):
        if s >= char_idx:
            return tok_idx
    return None


def find_assertion_token(
    offsets: list[tuple[int, int]],
    response: str,
    prompt_end_char: int,
    occurrence: str,
    token_pos: str,
) -> int | None:
    """Locate the assertion token in the full-text token sequence.

    occurrence: "last" (matches the verifier, which scores the final
        `<answer>`) or "first".
    token_pos:  "after" (the token immediately following the `<answer>` tag --
        the spec's definition) or "open" (the `<answer>` opening token itself,
        matching cache_answer_positions.py).
    """
    matches = list(_ANSWER_OPEN_RE.finditer(response))
    if not matches:
        return None
    m = matches[-1] if occurrence == "last" else matches[0]
    if token_pos == "after":
        char_in_full = prompt_end_char + m.end()
        return first_token_at_or_after(offsets, char_in_full)
    # token_pos == "open"
    char_in_full = prompt_end_char + m.start()
    tok = char_to_token_index(offsets, char_in_full)
    if tok is None:
        tok = first_token_at_or_after(offsets, char_in_full)
    return tok


# ---------------------------------------------------------------------------
# Probe loading. Honors the "bare sklearn LogisticRegression at probe.pkl"
# assumption, but transparently supports a Pipeline (scaler + LR) and the
# repo's .npz direction file (save_probe_vector.py) as conveniences.
# ---------------------------------------------------------------------------


def load_probe_scorer(path: str):
    """Return f(X: (N, D) float32) -> scores: (N,) float.

    For AUROC only the *ranking* of scores matters, so a constant offset
    (the LR intercept, or the .npz file's missing intercept) is irrelevant.
    """
    if path.endswith(".npz"):
        with np.load(path) as d:
            # Prefer the raw input-space direction; fall back to unit vector.
            v = d["v_input_raw"] if "v_input_raw" in d.files else d["v_unit"]
        v = np.asarray(v, dtype=np.float32)
        return lambda X: np.asarray(X, dtype=np.float32) @ v

    with open(path, "rb") as f:
        obj = pickle.load(f)
    # NOTE: if your LogisticRegression was trained on StandardScaler-transformed
    # features (as train_probe.py does), pickle a Pipeline(scaler, lr) so the
    # scaling is applied here. A bare LR is applied to raw hidden states as-is.
    # train_answer_probe.py pickles a Pipeline, so scaling travels with it.
    if hasattr(obj, "predict_proba"):
        return lambda X: obj.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    if hasattr(obj, "decision_function"):
        return lambda X: obj.decision_function(np.asarray(X, dtype=np.float32))
    raise ValueError(
        f"Loaded probe from {path} has neither predict_proba nor decision_function."
    )


# ---------------------------------------------------------------------------
# Extraction phase: read each problem's existing rollouts, forward-pass for the
# layer-L hidden state at the assertion token, apply the probe. No sampling.
# Cached to JSONL (one row per rollout).
# ---------------------------------------------------------------------------


def extract_and_score_checkpoint(
    model_path: str,
    ckpt_name: str,
    rows: list[dict],
    probe_path: str,
    args,
    cache_path: str,
) -> list[dict]:
    """`rows`: rollout rows (each with prompt/target/nums/response[]/scores[]
    and _orig_idx). Returns one record per rollout:
        {prompt_idx, orig_prompt_idx, resp_idx, label, score, probe_score|None}
    Writes them to `cache_path` (JSONL). If the cache is complete it is reused
    and neither the model nor the probe is loaded (so an analysis-only re-run
    from cached scores needs neither GPU nor probe.pkl).
    """
    if os.path.exists(cache_path):
        recs = [json.loads(l) for l in open(cache_path) if l.strip()]
        if len({r["prompt_idx"] for r in recs}) >= len(rows):
            print(f"[{ckpt_name}] reusing cache {cache_path} ({len(recs)} rollouts)",
                  flush=True)
            return recs
        print(f"[{ckpt_name}] cache {cache_path} incomplete; recomputing", flush=True)

    # Cache miss -> we actually extract, so load the probe + model now.
    scorer = load_probe_scorer(probe_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{ckpt_name}] loading model {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required (need offset_mapping).")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to("cuda").eval()

    records: list[dict] = []
    n_no_answer = 0
    out_f = open(cache_path, "w")
    for p_idx, row in enumerate(rows):
        prompt_text = row["prompt"]
        responses = row["response"][: args.n_rollouts]
        scores = row["scores"][: args.n_rollouts]
        prompt_end_char = len(prompt_text)
        orig_idx = int(row.get("_orig_idx", p_idx))

        for r_idx, (response, score) in enumerate(zip(responses, scores)):
            label = int(float(score) == 1.0)
            full_text = prompt_text + response
            enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True,
                            max_length=args.max_seq_len, return_tensors="pt")
            input_ids = enc["input_ids"].to("cuda")
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]

            tok_idx = find_assertion_token(
                offsets, response, prompt_end_char,
                args.answer_occurrence, args.answer_token_pos,
            )
            probe_score = None
            if tok_idx is not None and tok_idx < input_ids.shape[1]:
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_hidden_states=True)
                vec = out.hidden_states[args.layer][0, tok_idx].float().cpu().numpy()
                probe_score = float(scorer(vec[None, :])[0])
            else:
                n_no_answer += 1

            rec = {"prompt_idx": p_idx, "orig_prompt_idx": orig_idx, "resp_idx": r_idx,
                   "label": label, "score": float(score), "probe_score": probe_score}
            records.append(rec)
            out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if (p_idx + 1) % 25 == 0:
            print(f"[{ckpt_name}] {p_idx + 1}/{len(rows)} problems extracted", flush=True)
    out_f.close()

    del model
    torch.cuda.empty_cache()
    print(f"[{ckpt_name}] done: {len(records)} rollouts, "
          f"{n_no_answer} had no locatable <answer> token (excluded from AUROC).",
          flush=True)
    return records


# ---------------------------------------------------------------------------
# Analysis: per-problem AUROC + accuracy.
# ---------------------------------------------------------------------------


def per_problem_metrics(records: list[dict], n_problems: int):
    """Returns (auroc, accuracy) arrays of length n_problems (auroc may be NaN)."""
    from sklearn.metrics import roc_auc_score

    by_problem: dict[int, list[dict]] = {i: [] for i in range(n_problems)}
    for r in records:
        by_problem.setdefault(r["prompt_idx"], []).append(r)

    auroc = np.full(n_problems, np.nan)
    accuracy = np.full(n_problems, np.nan)
    for i in range(n_problems):
        recs = by_problem.get(i, [])
        if not recs:
            continue
        labels_all = [r["label"] for r in recs]
        accuracy[i] = float(np.mean(labels_all))  # all K rollouts

        # AUROC only over rollouts with a valid probe score.
        pairs = [(r["probe_score"], r["label"]) for r in recs
                 if r["probe_score"] is not None]
        if len(pairs) < 2:
            continue
        scores = np.array([p[0] for p in pairs], dtype=float)
        labels = np.array([p[1] for p in pairs], dtype=int)
        if len(set(labels.tolist())) < 2:
            continue  # all-correct or all-incorrect => AUROC undefined
        auroc[i] = float(roc_auc_score(labels, scores))
    return auroc, accuracy


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------


def make_plot(probe_drop, accuracy_delta, spearman_r, spearman_p, out_png, thr=0.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe_drop = np.asarray(probe_drop, dtype=float)
    accuracy_delta = np.asarray(accuracy_delta, dtype=float)
    valid = np.isfinite(probe_drop) & np.isfinite(accuracy_delta)

    colors = np.where(accuracy_delta > thr, "tab:green",
                      np.where(accuracy_delta < -thr, "tab:red", "tab:gray"))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.scatter(probe_drop[valid], accuracy_delta[valid],
               c=colors[valid], s=36, alpha=0.75, edgecolors="white", linewidths=0.4)

    # Regression line over the valid (correlation) points.
    if valid.sum() >= 2:
        x = probe_drop[valid]
        y = accuracy_delta[valid]
        b, a = np.polyfit(x, y, 1)  # y = b*x + a
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b * xs + a, color="black", lw=1.6, ls="--",
                label=f"fit: y = {b:.2f}x + {a:.2f}")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_xlabel("probe_drop  =  AUROC(SFT) - AUROC(RLOO)   (>0 : probe got worse)")
    ax.set_ylabel("accuracy_delta  =  acc(RLOO) - acc(SFT)")
    ax.set_title("Per-problem probe AUROC drop vs behavioral accuracy change")

    # Spearman annotation.
    def _fmt(v, sci=False):
        if v is None or math.isnan(v):
            return "n/a"
        return f"{v:.2e}" if sci else f"{v:.3f}"
    p_sci = spearman_p is not None and not math.isnan(spearman_p) and spearman_p < 1e-3
    p_txt = _fmt(spearman_p, sci=p_sci)
    r_txt = _fmt(spearman_r)
    ax.text(0.5, 0.99, f"Spearman r = {r_txt},  p = {p_txt}  (n={int(valid.sum())})",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    # Quadrant labels (corners, in axes coords).
    quad = dict(fontsize=8.5, alpha=0.65, style="italic")
    ax.text(0.02, 0.92, "probe stable\naccuracy improved",
            transform=ax.transAxes, ha="left", va="top", color="tab:green", **quad)
    ax.text(0.98, 0.92, "probe dropped\naccuracy improved\n(overconfidence)",
            transform=ax.transAxes, ha="right", va="top", color="tab:olive", **quad)
    ax.text(0.02, 0.06, "probe stable\naccuracy degraded",
            transform=ax.transAxes, ha="left", va="bottom", color="tab:gray", **quad)
    ax.text(0.98, 0.06, "probe dropped\naccuracy degraded\n(our hypothesis)",
            transform=ax.transAxes, ha="right", va="bottom", color="tab:red", **quad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)


# ---------------------------------------------------------------------------


def _nan_to_none(arr) -> list:
    return [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
            for x in arr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft_model", default="asingh15/qwen-sft-countdown-defaultproj",
                        help="Checkpoint that produced --sft_rollouts_json (for the "
                             "forward-pass hidden states).")
    parser.add_argument("--rloo_model",
                        default="/vol/checkpoints/rloo_checkpoints/rloo_training/"
                                "rloo_fixed_v2/latest_checkpoint/model",
                        help="Checkpoint that produced --rloo_rollouts_json.")
    parser.add_argument("--sft_rollouts_json",
                        default="/vol/evaluation/eval_results/eval_c_sft_n500.json",
                        help="Existing C_SFT rollouts (prompt/target/nums/response/scores).")
    parser.add_argument("--rloo_rollouts_json",
                        default="/vol/evaluation/eval_results/eval_c_outcome_n500.json",
                        help="Existing C_outcome rollouts, row-aligned to the SFT JSON.")
    parser.add_argument("--probe", default="probe.pkl",
                        help="sklearn Pipeline / LogisticRegression (.pkl) or "
                             "direction file (.npz from save_probe_vector.py).")
    parser.add_argument("--clean_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep only the contamination-filtered 'clean' problems "
                             "(matching the paper's clean-406). Use --no-clean_only for all.")
    parser.add_argument("--contam_json", default="extension/data/contaminated_prompt_idx.json",
                        help="JSON with 'clean'/'contaminated' row-index lists.")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--n_rollouts", "-K", type=int, default=16,
                        help="Cap on rollouts used per problem (uses the first K stored).")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--answer_occurrence", choices=("first", "last"), default="last",
                        help="Which <answer> block to probe. 'last' matches the "
                             "verifier, which scores the final <answer>.")
    parser.add_argument("--answer_token_pos", choices=("after", "open"), default="after",
                        help="'after': token immediately following <answer> (spec). "
                             "'open': the <answer> opening token (cache_answer_positions).")
    parser.add_argument("--acc_threshold", type=float, default=0.1,
                        help="|accuracy_delta| band for green/gray/red coloring.")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", default="extension/outputs/n500/figures")
    parser.add_argument("--cache_dir", default=None,
                        help="Where to cache per-rollout JSONL (default: output_dir).")
    args = parser.parse_args()

    out_dir = args.output_dir
    cache_dir = args.cache_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    sft_all = [json.loads(l) for l in open(args.sft_rollouts_json) if l.strip()]
    rloo_all = [json.loads(l) for l in open(args.rloo_rollouts_json) if l.strip()]
    for i, r in enumerate(sft_all):
        r["_orig_idx"] = i
    for i, r in enumerate(rloo_all):
        r["_orig_idx"] = i
    print(f"[main] {len(sft_all)} SFT / {len(rloo_all)} RLOO problem rows loaded", flush=True)

    # Candidate problems present in BOTH checkpoints' rollout sets.
    common = list(range(min(len(sft_all), len(rloo_all))))

    # Contamination filter on original problem index.
    set_tag = "all"
    if args.clean_only:
        if os.path.exists(args.contam_json):
            clean_set = set(json.load(open(args.contam_json))["clean"])
            kept = [i for i in common if i in clean_set]
            print(f"[main] clean filter: kept {len(kept)}/{len(common)} problems "
                  f"(dropped {len(common) - len(kept)} RLOO-train-contaminated)", flush=True)
            common = kept
            set_tag = f"clean{len(common)}"
        else:
            print(f"[main] WARNING: --clean_only set but {args.contam_json} missing; "
                  f"using all {len(common)} problems", flush=True)
    if args.max_prompts is not None:
        common = common[: args.max_prompts]
        set_tag += f"_first{len(common)}"

    # Aligned per-checkpoint row lists: position j -> original problem common[j].
    sft_rows = [sft_all[i] for i in common]
    rloo_rows = [rloo_all[i] for i in common]
    n_problems = len(common)
    print(f"[main] {n_problems} problems, K={args.n_rollouts}, layer L{args.layer}, "
          f"probe={args.probe}", flush=True)

    recs_sft = extract_and_score_checkpoint(
        args.sft_model, "C_SFT", sft_rows, args.probe, args,
        os.path.join(cache_dir, f"rollout_scores_C_SFT_{set_tag}.jsonl"))
    recs_rloo = extract_and_score_checkpoint(
        args.rloo_model, "C_outcome", rloo_rows, args.probe, args,
        os.path.join(cache_dir, f"rollout_scores_C_outcome_{set_tag}.jsonl"))

    auroc_sft, acc_sft = per_problem_metrics(recs_sft, n_problems)
    auroc_rloo, acc_rloo = per_problem_metrics(recs_rloo, n_problems)

    accuracy_delta = acc_rloo - acc_sft                 # Step 2
    probe_drop = auroc_sft - auroc_rloo                 # Step 3 (NaN if either NaN)

    # Step 4: Spearman over problems with a defined probe_drop AND accuracy_delta.
    from scipy.stats import spearmanr
    valid = np.isfinite(probe_drop) & np.isfinite(accuracy_delta)
    n_valid = int(valid.sum())
    if n_valid >= 2:
        spearman_r, spearman_p = spearmanr(probe_drop[valid], accuracy_delta[valid])
        spearman_r, spearman_p = float(spearman_r), float(spearman_p)
    else:
        spearman_r, spearman_p = float("nan"), float("nan")
    print(f"[main] Spearman r={spearman_r:.4f} p={spearman_p:.3e} "
          f"over n={n_valid}/{n_problems} problems with defined AUROC on both ckpts",
          flush=True)

    out_png = os.path.join(out_dir, "probe_behavioral_correlation.png")
    make_plot(probe_drop, accuracy_delta, spearman_r, spearman_p,
              out_png, thr=args.acc_threshold)

    # Step 5: JSON (NaN -> null for valid JSON).
    out_json = os.path.join(out_dir, "probe_behavioral_correlation.json")
    payload = {
        "spearman_r": None if math.isnan(spearman_r) else spearman_r,
        "spearman_p": None if math.isnan(spearman_p) else spearman_p,
        "probe_drop": _nan_to_none(probe_drop),
        "accuracy_delta": _nan_to_none(accuracy_delta),
        "auroc_sft": _nan_to_none(auroc_sft),
        "auroc_rloo": _nan_to_none(auroc_rloo),
        # provenance / diagnostics (extra keys beyond the required six)
        "_meta": {
            "n_problems": n_problems,
            "n_valid_for_correlation": n_valid,
            "n_rollouts_per_problem": args.n_rollouts,
            "layer": args.layer,
            "answer_occurrence": args.answer_occurrence,
            "answer_token_pos": args.answer_token_pos,
            "acc_threshold": args.acc_threshold,
            "clean_only": bool(args.clean_only),
            "sft_rollouts_json": args.sft_rollouts_json,
            "rloo_rollouts_json": args.rloo_rollouts_json,
            "orig_prompt_idx": [int(i) for i in common],
            "sft_model": args.sft_model,
            "rloo_model": args.rloo_model,
            "probe": args.probe,
            "accuracy_sft": _nan_to_none(acc_sft),
            "accuracy_rloo": _nan_to_none(acc_rloo),
        },
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[main] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
