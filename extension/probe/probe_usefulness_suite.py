"""Bundled probe-usefulness suite — six experiments from one (or two) GPU forward passes.

Experiments in decreasing order of importance:

  #1  Selective abstention curve
      Probe score on the first rollout → commit if >= T, else abstain.
      Sweeps T; plots accuracy-on-attempted vs. coverage.

  #2  Verifier-free accuracy estimate
      Mean probe across rollouts per problem ≈ true pass rate.
      Reports |estimate − true| in pp (no answer key needed).

  #3  Probe-scaling curve  (K = 1, 2, 4, 8, 16)
      probe-best-of-K, random-pick-of-K, oracle-pass@K as K grows.

  #4  Probe vs self-consistency (majority vote)
      Compare probe-best, majority-vote, intersection hybrid.
      Disaggrement analysis: when they differ, who is right?

  #5  Train-once transfer  (probe trained on C_SFT → selects C_outcome)
      Second forward pass on C_SFT rollouts; train full probe; apply to
      C_outcome. Tests whether the probe ports across policy checkpoints.
      (Skip with --skip_transfer to save ~15 min on Modal.)

  #6  Probe-weighted voting
      Each rollout's answer-vote is weighted by its probe score.
      Does it beat plain majority vote?

Experiments #1–4 and #6 share one forward pass over C_outcome rollouts.
Experiment #5 adds one forward pass over C_SFT rollouts.

Usage:
    modal run modal_train.py probe_usefulness_suite
    python extension/probe/probe_usefulness_suite.py       # local (GPU needed)

-------------------------------------------------------------------------------
READ BEFORE QUOTING THESE NUMBERS  (see CODE_AUDIT.md)

This file produces `probe_usefulness_suite_results_n406.json`, which supplies
several README headline numbers. It was deleted in cd8cd5e and restored, because
those numbers were still being cited with no way to regenerate them. Its logic is
unchanged from the version that produced the committed JSON -- deliberately, so
the published numbers stay reproducible. Two known gaps, neither patched here:

  * Exp #3 (probe-scaling / best-of-K) has NO structural baseline. Most of the
    best-of-K lift on this policy is available from "pick the rollout with the
    shortest <think> body", which needs no probe and no forward pass: on
    C_outcome clean-406 that is +8.5 pp of the probe's +11.7 pp. Run
    `extension/probe/structural_baselines.py` and report the two side by side.

  * Exp #2 (verifier-free accuracy estimate) is largely a logistic-regression
    calibration identity: an intercept-fitted LR has mean predicted probability
    equal to the base rate. A ZERO-SIGNAL probe on noise features already lands
    within ~1.4 pp of true accuracy on this data. The real probe's ~0.4 pp is an
    improvement on that, but the comparison must be against the noise floor, not
    against zero. `probe_as_eval_proxy.py` now runs that null control inline.

Exp #1's thresholds target a fixed coverage (not maximised accuracy) and Exp #4
votes on the answer string, so neither has the threshold-tuning or
oracle-majority problem flagged elsewhere in the audit.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from extension.training.probe_reward_rloo import _load_frozen_model, _think_close_hidden

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _normalize_eq(eq: str) -> str:
    return "".join(eq.strip().split())


def _first_eq(resp: str) -> str | None:
    m = _ANSWER_RE.search(resp)
    return _normalize_eq(m.group(1)) if m else None


def _is_correct(eq: str, target: int, nums: list) -> bool:
    try:
        ns = [int(n) for n in re.findall(r"\d+", eq.strip())]
        if sorted(ns) != sorted(int(x) for x in nums):
            return False
        if not re.match(r"^[\d+\-*/().\s]+$", eq.strip()):
            return False
        result = eval(eq.strip(), {"__builtins__": None}, {})  # noqa: S307
        return abs(result - int(target)) < 1e-5
    except Exception:
        return False


def _first_block_correct(resp: str, target: int, nums: list) -> int:
    m = _ANSWER_RE.search(resp)
    if not m:
        return 0
    return int(_is_correct(m.group(1), target, nums))


def _gather(tok, model, rows: list, layer: int, max_responses: int, tag: str):
    """Forward-pass all rollouts in pre-filtered rows.

    Skips rollouts with no </think> token (hidden state is None).
    Returns X (N, d), y (N,), groups (N,), eqs list[str|None].
    Groups are 0-indexed by position in `rows` (not original prompt_idx).
    """
    Xl: list[np.ndarray] = []
    yl: list[int] = []
    gl: list[int] = []
    eq_l: list[str | None] = []
    n_total = 0
    for pidx, row in enumerate(rows):
        target = int(row["target"])
        nums = list(row["nums"])
        prompt = row["prompt"]
        for resp in row["response"][:max_responses]:
            n_total += 1
            h = _think_close_hidden(tok, model, prompt, resp, layer)
            if h is None:
                continue
            Xl.append(h)
            yl.append(_first_block_correct(resp, target, nums))
            gl.append(pidx)
            eq_l.append(_first_eq(resp))
        if (pidx + 1) % 50 == 0:
            print(f"[suite/{tag}]  {pidx + 1}/{len(rows)} problems, "
                  f"{len(Xl)}/{n_total} valid (have </think>)", flush=True)
    print(f"[suite/{tag}]  done: {len(rows)} problems, "
          f"{len(Xl)}/{n_total} usable rollouts", flush=True)
    X = np.asarray(Xl, np.float32)
    y = np.asarray(yl, int)
    groups = np.asarray(gl)
    return X, y, groups, eq_l


def _held_out_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """GroupKFold(5) cross-validated LR probe. Returns (scores, auroc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scores = np.full(len(y), np.nan)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(set(y.tolist())) < 2:
        return scores, float("nan")

    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        pipe = Pipeline([("s", StandardScaler()),
                         ("l", LogisticRegression(C=0.1, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    mask = ~np.isnan(scores)
    if len(set(y[mask].tolist())) < 2:
        return scores, float("nan")
    return scores, float(roc_auc_score(y[mask], scores[mask]))


def _make_full_pipe(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("s", StandardScaler()),
                     ("l", LogisticRegression(C=0.1, max_iter=2000))])
    pipe.fit(X, y)
    return pipe


def _group_by_problem(groups: np.ndarray, scores: np.ndarray,
                      y: np.ndarray, eqs: list) -> dict[int, list]:
    """Returns {pidx: [(flat_idx, score, label, eq)]} sorted by flat_idx."""
    by_p: dict[int, list] = defaultdict(list)
    for i, (pidx, sc, lab, eq) in enumerate(
            zip(groups.tolist(), scores.tolist(), y.tolist(), eqs)):
        if sc != sc:  # isnan via nan != nan
            continue
        by_p[int(pidx)].append((i, float(sc), int(lab), eq))
    for p in by_p:
        by_p[p].sort()
    return by_p


def _save_fig(fig, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def _init_wandb(args):
    """Best-effort W&B init. Returns the run or None (never raises)."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[suite] wandb not installed; skipping W&B logging.", flush=True)
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "model": args.model,
                "layer": args.layer,
                "max_responses": args.max_responses,
                "clean_only": args.clean_only,
                "skip_transfer": args.skip_transfer,
                "outcome_json": args.outcome_json,
                "sft_json": args.sft_json,
            },
        )
        print(f"[suite] W&B logging to {run.url}", flush=True)
        return run
    except Exception as e:  # noqa: BLE001
        print(f"[suite] wandb.init failed ({e}); continuing without W&B.", flush=True)
        return None


def _flatten_scalars(obj, prefix: str, out: dict) -> None:
    """Recursively collect finite scalar leaves into out[dot/path] = value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten_scalars(v, key, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        if isinstance(obj, float) and obj != obj:  # nan
            return
        out[prefix] = obj


def _log_wandb(run, summary: dict, out_dir: str) -> None:
    """Log every scalar metric (incl. verifier ground-truth), the scaling /
    transfer curves as line plots, and the saved figures."""
    if run is None:
        return
    import wandb

    # 1) All scalar metrics, flattened (covers exp1-6 + probe_auroc).
    flat: dict = {}
    _flatten_scalars(summary, "", flat)

    # 2) Explicit verifier-derived ground-truth namespace for clarity.
    e2 = summary.get("exp2") or {}
    e1 = summary.get("exp1") or {}
    if "true_acc" in e2:
        flat["verifier/true_acc"] = e2["true_acc"]
    if "base_acc" in e1:
        flat["verifier/base_pass@1"] = e1["base_acc"]
    for K, v in (summary.get("exp3") or {}).items():
        if isinstance(v, dict) and v.get("oracle") is not None:
            flat[f"verifier/oracle_pass@{K}"] = v["oracle"]

    run.summary.update(flat)
    run.log(flat)

    # 3) Scaling curve as a line plot (probe vs random vs oracle).
    e3 = summary.get("exp3") or {}
    if e3:
        ks = sorted(int(k) for k in e3.keys())
        try:
            run.log({"exp3/scaling_curve": wandb.plot.line_series(
                xs=ks,
                ys=[[e3[str(k)]["probe"] for k in ks],
                    [e3[str(k)]["random"] for k in ks],
                    [e3[str(k)]["oracle"] for k in ks]],
                keys=["probe-best-of-K", "random-pick", "oracle pass@K"],
                title="Exp #3: Probe-scaling curve", xname="K")})
        except Exception as e:  # noqa: BLE001
            print(f"[suite] scaling line_series skipped: {e}", flush=True)

    # 4) Transfer curve (exp5) as a line plot, if it ran.
    e5 = summary.get("exp5") or {}
    bok = (e5 or {}).get("transfer_bestofK") if e5 else None
    if bok:
        ks = sorted(int(k) for k in bok.keys())
        try:
            run.log({"exp5/transfer_curve": wandb.plot.line_series(
                xs=ks, ys=[[bok[str(k)] for k in ks]],
                keys=["transfer-probe-best-of-K"],
                title="Exp #5: Train-once transfer best-of-K", xname="K")})
        except Exception as e:  # noqa: BLE001
            print(f"[suite] transfer line_series skipped: {e}", flush=True)

    # 5) Figures.
    for fname in ("fig1_abstention.png", "fig3_scaling.png", "fig4_vs_majority.png"):
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            run.log({f"figures/{fname[:-4]}": wandb.Image(fpath)})

    run.finish()
    print("[suite] W&B logging complete.", flush=True)


def _nan_to_none(obj):
    """Recursively make dicts/lists JSON-safe (nan→null, numpy→python)."""
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, np.floating):
        return None if obj != obj else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Experiment #1 — Selective abstention
# ---------------------------------------------------------------------------


def exp1_abstention(by_p: dict, prompts: list,
                    auroc: float, out_dir: str) -> dict:
    print("\n=== Exp #1: Selective abstention ===", flush=True)
    first_probe = {p: by_p[p][0][1] for p in prompts if by_p[p]}
    first_label = {p: by_p[p][0][2] for p in prompts if by_p[p]}
    n = len(prompts)
    base_acc = float(np.mean(list(first_label.values()))) if first_label else float("nan")

    sweep = []
    for T in np.linspace(0.0, 1.0, 101):
        attempted = [p for p in prompts if first_probe.get(p, -1.0) >= T]
        if not attempted:
            sweep.append({"T": float(T), "cov": 0.0, "acc": None, "n": 0})
        else:
            sweep.append({
                "T": float(T),
                "cov": len(attempted) / n,
                "acc": float(np.mean([first_label[p] for p in attempted])),
                "n": len(attempted),
            })

    valid = [r for r in sweep if r["acc"] is not None]
    print(f"  baseline (no abstention): {base_acc:.4f}")
    for tgt in [0.9, 0.7, 0.5, 0.3]:
        r = min(valid, key=lambda x: abs(x["cov"] - tgt))
        print(f"  coverage≈{tgt:.2f}: T={r['T']:.2f}, n={r['n']}, acc={r['acc']:.4f}")

    r30 = min(valid, key=lambda x: abs(x["cov"] - 0.3))
    r50 = min(valid, key=lambda x: abs(x["cov"] - 0.5))
    print(f"\n  [HEADLINE #1] At coverage={r30['cov']*100:.0f}%: "
          f"acc={r30['acc']:.3f} (base {base_acc:.3f}, "
          f"+{(r30['acc'] - base_acc)*100:.1f} pp)", flush=True)

    cov_arr = [r["cov"] for r in sweep]
    acc_arr = [r["acc"] if r["acc"] is not None else float("nan") for r in sweep]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
    ax.plot(cov_arr, acc_arr, lw=2, color="#3a6dba", marker="o", markersize=3)
    ax.axhline(base_acc, color="grey", ls="--", lw=1,
               label=f"baseline acc={base_acc:.3f}")
    ax.set_xlabel("coverage (fraction of problems attempted)")
    ax.set_ylabel("accuracy on attempted problems")
    ax.set_title(f"Exp #1: Probe-guided selective abstention\n(AUROC={auroc:.3f})")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig1_abstention.png"))

    # Keep only representative sweep points in results (not all 101)
    summary_pts = {f"coverage_{int(tgt*100)}pct": min(valid, key=lambda x: abs(x["cov"] - tgt))
                   for tgt in [0.9, 0.7, 0.5, 0.3]}
    return {"base_acc": base_acc, "at_30pct": r30, "at_50pct": r50,
            "summary_pts": summary_pts}


# ---------------------------------------------------------------------------
# Experiment #2 — Verifier-free accuracy estimate
# ---------------------------------------------------------------------------


def exp2_eval_proxy(by_p: dict, prompts: list) -> dict:
    print("\n=== Exp #2: Verifier-free accuracy estimate ===", flush=True)
    per_p_true = np.array([np.mean([t[2] for t in by_p[p]]) for p in prompts])
    per_p_probe = np.array([np.mean([t[1] for t in by_p[p]]) for p in prompts])
    true_acc = float(np.mean(per_p_true))
    est_mean = float(np.mean(per_p_probe))
    est_vote = float(np.mean(per_p_probe >= 0.5))
    err_mean_pp = abs(est_mean - true_acc) * 100
    err_vote_pp = abs(est_vote - true_acc) * 100
    print(f"  True dataset accuracy (verifier):        {true_acc:.4f}")
    print(f"  Probe-estimated (mean probe):            {est_mean:.4f}  "
          f"(err={err_mean_pp:.2f} pp)")
    print(f"  Probe-estimated (probe>=0.5 vote):       {est_vote:.4f}  "
          f"(err={err_vote_pp:.2f} pp)")
    print(f"\n  [HEADLINE #2] Probe estimates accuracy to ±{err_mean_pp:.1f} pp "
          "with no answer key", flush=True)
    return {"true_acc": true_acc, "est_mean": est_mean, "est_vote": est_vote,
            "err_mean_pp": err_mean_pp, "err_vote_pp": err_vote_pp}


# ---------------------------------------------------------------------------
# Experiment #3 — Probe-scaling curve
# ---------------------------------------------------------------------------


def exp3_scaling(by_p: dict, prompts: list, out_dir: str,
                 Ks: tuple = (1, 2, 4, 8, 16)) -> dict:
    print(f"\n=== Exp #3: Probe-scaling curve (Ks={list(Ks)}) ===", flush=True)
    results: dict[int, dict] = {}
    for K in Ks:
        probe_k: list[float] = []
        rand_k: list[float] = []
        oracle_k: list[float] = []
        for p in prompts:
            v = by_p[p][:K]
            if not v:
                continue
            probe_k.append(float(max(v, key=lambda t: t[1])[2]))
            rand_k.append(float(np.mean([t[2] for t in v])))
            oracle_k.append(1.0 if any(t[2] for t in v) else 0.0)
        results[K] = {
            "probe": float(np.mean(probe_k)),
            "random": float(np.mean(rand_k)),
            "oracle": float(np.mean(oracle_k)),
        }
        print(f"  K={K:2d}  probe={results[K]['probe']:.4f}  "
              f"rand={results[K]['random']:.4f}  "
              f"oracle={results[K]['oracle']:.4f}", flush=True)

    # Headline: probe-best-of-4 ≈ random-of-16?
    p4 = results.get(4, {}).get("probe", float("nan"))
    p16 = results.get(16, {}).get("probe", float("nan"))
    r16 = results.get(16, {}).get("random", float("nan"))
    print(f"\n  [HEADLINE #3] probe-best-of-4={p4:.3f}  random-of-16={r16:.3f}  "
          f"probe-best-of-16={p16:.3f}", flush=True)

    ks = sorted(Ks)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(ks, [results[K]["probe"]  for K in ks], "o-", color="#3a6dba",
            lw=2, label="probe-best-of-K")
    ax.plot(ks, [results[K]["random"] for K in ks], "s--", color="#888888",
            lw=1.5, label="random-pick (expected)")
    ax.plot(ks, [results[K]["oracle"] for K in ks], "^--", color="#c45252",
            lw=1.5, label="oracle pass@K")
    ax.set_xlabel("K (rollouts per problem)")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_title("Exp #3: Probe-scaling curve (probe-best-of-K)")
    ax.legend()
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig3_scaling.png"))

    return {str(K): v for K, v in results.items()}


# ---------------------------------------------------------------------------
# Experiment #4 — Probe vs self-consistency (majority vote)
# ---------------------------------------------------------------------------


def exp4_majority(by_p: dict, prompts: list, out_dir: str, K: int = 16) -> dict:
    print(f"\n=== Exp #4: Probe vs majority vote (K={K}) ===", flush=True)
    probe_acc: list[int] = []
    maj_acc: list[int] = []
    inter_acc: list[int] = []
    T_inter = 0.5
    n_agree = 0
    pb_dis: list[int] = []
    mb_dis: list[int] = []

    for p in prompts:
        v = by_p[p][:K]
        if not v:
            continue
        best = max(v, key=lambda t: t[1])
        probe_acc.append(best[2])

        counts = Counter(t[3] for t in v if t[3] is not None)
        if not counts:
            maj_acc.append(0)
            inter_acc.append(best[2])
            continue
        top_eq = counts.most_common(1)[0][0]
        maj_label = next((t[2] for t in v if t[3] == top_eq), 0)
        maj_acc.append(maj_label)

        eq_probes = [t[1] for t in v if t[3] == top_eq]
        inter_acc.append(maj_label if float(np.mean(eq_probes)) >= T_inter else best[2])

        if best[3] == top_eq:
            n_agree += 1
        else:
            pb_dis.append(best[2])
            mb_dis.append(maj_label)

    pa = float(np.mean(probe_acc)) if probe_acc else float("nan")
    ma = float(np.mean(maj_acc)) if maj_acc else float("nan")
    ia = float(np.mean(inter_acc)) if inter_acc else float("nan")
    n = len(prompts)
    pb_wins = sum(1 for a, b in zip(pb_dis, mb_dis) if a > b)
    mb_wins = sum(1 for a, b in zip(pb_dis, mb_dis) if b > a)

    print(f"  probe-best-of-{K}:          {pa:.4f}")
    print(f"  majority-vote-of-{K}:       {ma:.4f}")
    print(f"  intersection hybrid:        {ia:.4f}")
    print(f"  agreement rate: {n_agree}/{n} = {100*n_agree/n:.1f}%")
    if pb_dis:
        ratio = (pb_wins / mb_wins) if mb_wins > 0 else float("inf")
        print(f"  On {len(pb_dis)} disagreements: probe wins {pb_wins}, "
              f"majority wins {mb_wins}")
        print(f"\n  [HEADLINE #4] When probe and majority disagree, "
              f"probe is right {ratio:.1f}× more than majority", flush=True)

    base = float(np.mean([by_p[p][0][2] for p in prompts if by_p[p]]))
    methods = [f"probe-best\nof-{K}", f"majority\nof-{K}", f"intersection\nhybrid"]
    vals = [pa, ma, ia]
    colors = ["#3a6dba", "#888888", "#3a8b2f"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    bars = ax.bar(methods, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=11)
    ax.axhline(base, color="red", ls="--", lw=1, label=f"pass@1={base:.3f}")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Exp #4: Probe vs majority vote (K={K})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig4_vs_majority.png"))

    return {
        "probe": pa, "majority": ma, "intersection": ia,
        "agree_rate": n_agree / n if n else 0.0,
        "n_disagree": len(pb_dis),
        "probe_wins": pb_wins,
        "maj_wins": mb_wins,
    }


# ---------------------------------------------------------------------------
# Experiment #5 — Train-once transfer (C_SFT probe → C_outcome)
# ---------------------------------------------------------------------------


def exp5_transfer(tok, model, sft_rows: list,
                  X_out: np.ndarray, y_out: np.ndarray,
                  groups_out: np.ndarray, eqs_out: list,
                  prompts_out: list, args) -> dict | None:
    print("\n=== Exp #5: Train-once transfer (C_SFT → C_outcome) ===", flush=True)

    Xs, ys, gs, _ = _gather(tok, model, sft_rows, args.layer,
                              args.max_responses, "C_SFT")
    print(f"  C_SFT: {len(ys)} usable rollouts, acc={ys.mean():.3f}", flush=True)
    if len(set(ys.tolist())) < 2:
        print("  Skipping: C_SFT rollouts have only one label class.", flush=True)
        return None

    _, sft_auroc = _held_out_scores(Xs, ys, gs)
    print(f"  C_SFT held-out AUROC: {sft_auroc:.3f}", flush=True)

    # Train full probe on ALL C_SFT rollouts (no CV — this is the transfer probe)
    transfer_pipe = _make_full_pipe(Xs, ys)

    # Score C_outcome rollouts with the transfer probe
    transfer_scores = transfer_pipe.predict_proba(X_out)[:, 1]

    by_p_t: dict[int, list] = defaultdict(list)
    for i, (pidx, sc, lab, eq) in enumerate(
            zip(groups_out.tolist(), transfer_scores.tolist(),
                y_out.tolist(), eqs_out)):
        by_p_t[int(pidx)].append((i, float(sc), int(lab), eq))
    for p in by_p_t:
        by_p_t[p].sort()

    Ks = [1, 2, 4, 8, 16]
    transfer_bok: dict[int, float] = {}
    for K in Ks:
        acc = []
        for p in prompts_out:
            v = by_p_t.get(p, [])[:K]
            if not v:
                continue
            acc.append(float(max(v, key=lambda t: t[1])[2]))
        transfer_bok[K] = float(np.mean(acc)) if acc else float("nan")
        print(f"  transfer-probe-best-of-{K}: {transfer_bok[K]:.4f}", flush=True)

    t16 = transfer_bok.get(16, float("nan"))
    print(f"\n  [HEADLINE #5] Transfer probe (trained once on C_SFT) achieves "
          f"best-of-16={t16:.3f} on C_outcome (no per-checkpoint retraining)", flush=True)

    return {
        "sft_auroc": sft_auroc,
        "transfer_bestofK": {str(K): v for K, v in transfer_bok.items()},
    }


# ---------------------------------------------------------------------------
# Experiment #6 — Probe-weighted voting
# ---------------------------------------------------------------------------


def exp6_weighted_vote(by_p: dict, prompts: list, K: int = 16) -> dict:
    print(f"\n=== Exp #6: Probe-weighted voting (K={K}) ===", flush=True)
    weighted_acc: list[int] = []
    plain_acc: list[int] = []

    for p in prompts:
        v = by_p[p][:K]
        if not v:
            continue
        counts = Counter(t[3] for t in v if t[3] is not None)
        if not counts:
            weighted_acc.append(0)
            plain_acc.append(0)
            continue

        # Plain majority vote
        top_eq_plain = counts.most_common(1)[0][0]
        plain_acc.append(next((t[2] for t in v if t[3] == top_eq_plain), 0))

        # Probe-weighted vote: sum probe scores per unique equation
        eq_weight: dict[str, float] = defaultdict(float)
        eq_label: dict[str, int] = {}
        for _, sc, lab, eq in v:
            if eq is None:
                continue
            eq_weight[eq] += sc
            eq_label[eq] = lab
        if not eq_weight:
            weighted_acc.append(0)
            continue
        top_eq_w = max(eq_weight, key=eq_weight.__getitem__)
        weighted_acc.append(eq_label[top_eq_w])

    wa = float(np.mean(weighted_acc)) if weighted_acc else float("nan")
    pa = float(np.mean(plain_acc)) if plain_acc else float("nan")
    delta_pp = (wa - pa) * 100

    print(f"  plain majority-of-{K}:    {pa:.4f}")
    print(f"  probe-weighted-of-{K}:    {wa:.4f}  ({delta_pp:+.1f} pp vs majority)")
    beats = (wa > pa) and not (wa != wa)
    print(f"\n  [HEADLINE #6] Probe-weighted voting "
          f"{'beats' if beats else 'does NOT beat'} plain majority "
          f"({abs(delta_pp):.1f} pp {'gain' if beats else 'loss'})", flush=True)

    return {"plain_majority": pa, "probe_weighted": wa, "delta_pp": delta_pp}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="asingh15/qwen-sft-countdown-defaultproj")
    ap.add_argument("--outcome_json",  default="eval_c_outcome_n500.json")
    ap.add_argument("--sft_json",      default="eval_c_sft_n500.json")
    ap.add_argument("--contam_json",   default="extension/data/contaminated_prompt_idx.json")
    ap.add_argument("--clean_only",    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--layer",         type=int, default=16)
    ap.add_argument("--max_responses", type=int, default=16)
    ap.add_argument("--skip_transfer", action="store_true",
                    help="Skip Exp #5 (train-once transfer) to save ~15 min on Modal.")
    ap.add_argument("--out_dir",       default="/vol/outputs/probe_usefulness_suite")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                    help="Log all metrics + figures to Weights & Biases (default on).")
    ap.add_argument("--wandb_project", default="probe_usefulness_suite")
    ap.add_argument("--wandb_name",    default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)

    wandb_run = _init_wandb(args)

    def _load_clean(path: str) -> list:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if args.clean_only and os.path.exists(args.contam_json):
            clean = set(json.load(open(args.contam_json))["clean"])
            rows = [r for i, r in enumerate(rows) if i in clean]
        return rows

    outcome_rows = _load_clean(args.outcome_json)
    print(f"[suite] C_outcome: {len(outcome_rows)} clean problems  model={args.model}",
          flush=True)

    tok, model_obj = _load_frozen_model(args.model)

    # ---- Forward pass: C_outcome ----
    print(f"\n[suite] Forward pass: C_outcome "
          f"({len(outcome_rows)} probs × ≤{args.max_responses} rollouts)…", flush=True)
    X, y, groups, eqs = _gather(
        tok, model_obj, outcome_rows, args.layer, args.max_responses, "C_outcome")
    print(f"[suite] C_outcome: {len(y)} usable rollouts, acc={y.mean():.3f}", flush=True)

    # ---- Held-out probe scores (shared by #1-4, #6) ----
    print("\n[suite] Computing held-out probe scores (GroupKFold-5)…", flush=True)
    scores, auroc = _held_out_scores(X, y, groups)
    print(f"[suite] Held-out probe AUROC: {auroc:.3f}", flush=True)

    by_p = _group_by_problem(groups, scores, y, eqs)
    prompts = sorted(by_p.keys())
    n_prompts = len(prompts)
    print(f"[suite] {n_prompts} problems with ≥1 scored rollout", flush=True)

    summary: dict = {
        "model": args.model,
        "n_problems": n_prompts,
        "probe_auroc": auroc,
    }

    # ---- Run experiments in decreasing order of importance ----
    summary["exp1"] = exp1_abstention(by_p, prompts, auroc, args.out_dir)
    summary["exp2"] = exp2_eval_proxy(by_p, prompts)
    summary["exp3"] = exp3_scaling(by_p, prompts, args.out_dir)
    summary["exp4"] = exp4_majority(by_p, prompts, args.out_dir)

    if args.skip_transfer:
        print("\n[suite] Skipping Exp #5 (--skip_transfer).", flush=True)
        summary["exp5"] = None
    else:
        sft_rows = _load_clean(args.sft_json)
        print(f"[suite] C_SFT: {len(sft_rows)} clean problems", flush=True)
        summary["exp5"] = exp5_transfer(
            tok, model_obj, sft_rows, X, y, groups, eqs, prompts, args)

    summary["exp6"] = exp6_weighted_vote(by_p, prompts)

    # ---- Persist ----
    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(_nan_to_none(summary), f, indent=2, default=str)
    print(f"\n[suite] wrote {out_json}", flush=True)

    # ---- W&B logging (all metrics incl. verifier + curves + figures) ----
    _log_wandb(wandb_run, _nan_to_none(summary), args.out_dir)

    # ---- Headline summary ----
    print("\n" + "=" * 60, flush=True)
    print("HEADLINE SUMMARY", flush=True)
    print("=" * 60, flush=True)

    def _fmt(v, fmt=".3f"):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    e1 = summary.get("exp1") or {}
    r30 = e1.get("at_30pct") or {}
    print(f"  #1 Abstention : acc={_fmt(r30.get('acc'))} at "
          f"coverage={_fmt(r30.get('cov', 0), '.0%')} "
          f"(base {_fmt(e1.get('base_acc'))})", flush=True)

    e2 = summary.get("exp2") or {}
    print(f"  #2 Eval proxy : ±{_fmt(e2.get('err_mean_pp', float('nan')), '.1f')} pp "
          "estimate error", flush=True)

    e3 = summary.get("exp3") or {}
    r16_3 = e3.get("16") or {}
    print(f"  #3 Scaling    : BoK16={_fmt(r16_3.get('probe'))}  "
          f"rand={_fmt(r16_3.get('random'))}  "
          f"oracle={_fmt(r16_3.get('oracle'))}", flush=True)

    e4 = summary.get("exp4") or {}
    print(f"  #4 vs Majority: probe={_fmt(e4.get('probe'))}  "
          f"majority={_fmt(e4.get('majority'))}  "
          f"(probe wins {e4.get('probe_wins', '?')} vs "
          f"maj {e4.get('maj_wins', '?')} on disagreements)", flush=True)

    e5 = summary.get("exp5") or {}
    if e5:
        t16 = (e5.get("transfer_bestofK") or {}).get("16", float("nan"))
        print(f"  #5 Transfer   : BoK16={_fmt(t16)}  "
              f"(C_SFT AUROC={_fmt(e5.get('sft_auroc'))})", flush=True)

    e6 = summary.get("exp6") or {}
    print(f"  #6 WeightedVote: probe-weighted={_fmt(e6.get('probe_weighted'))}  "
          f"plain-majority={_fmt(e6.get('plain_majority'))}  "
          f"(delta={_fmt(e6.get('delta_pp', float('nan')), '+.1f')} pp)", flush=True)


if __name__ == "__main__":
    main()
