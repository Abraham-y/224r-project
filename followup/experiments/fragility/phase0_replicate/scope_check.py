"""Zero-cost scoping check: is there a phenomenon here worth harvesting for?

Runs on rollout TEXT alone — the post-Goodhart eval JSONs already on disk — so it
costs nothing and needs no GPU. It cannot answer H1-vs-H2 (that needs
activations), but it can kill or support the project's premises cheaply.

Three go/no-go questions:

  Q1  Is the collapse actually in the data we hold? If the post-RL eval JSONs
      don't show the published accuracy drop, the artefacts are wrong and no
      amount of harvesting will help.

  Q2  Does the H0 confound story survive contact with the post-RL rollouts?
      H0 predicts the template becomes MORE prevalent while its correlation with
      correctness COLLAPSES — the policy kept the style and dropped the
      substance. If instead the template correlation holds up post-RL, H0 is
      weakened before any GPU time is spent.

  Q3  Is there enough label variance left to fit anything? A checkpoint where
      accuracy has collapsed to ~7% may have too few positives for a retrained
      probe to be estimable. That directly bounds which checkpoints are worth
      harvesting, and it is better known now than after paying for them.

    python experiments/fragility/phase0_replicate/scope_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import labels as labels_mod, metrics_io, paths, probes  # noqa: E402

# label -> (eval json, what it is)
CHECKPOINTS = [
    ("C_SFT", "eval_c_sft_n500.json", "no RL (runB's init)"),
    ("C_outcome", "eval_c_outcome_n500.json", "vanilla verifier RL (runA's init)"),
    ("runA_postRL", "eval_runA_postRL_n500.json", "probe-as-reward, init C_outcome"),
    ("runB_postRL", "eval_runB_postRL_n500.json", "probe-as-reward, init C_SFT"),
]


def analyse(label: str, path: Path, note: str) -> dict | None:
    if not path.exists():
        print(f"[scope] {label}: {path.name} MISSING; row omitted")
        return None
    lab = labels_mod.build_label_table(path)
    clean = labels_mod.clean_prompt_idx()
    if clean is not None:
        lab = lab[lab["prompt_idx"].isin(clean)].reset_index(drop=True)

    y = lab["last_block"].to_numpy().astype(int)
    yf = lab["first_block"].to_numpy().astype(int)
    ts = lab["template_score"].to_numpy(dtype=float)

    rec = {
        "checkpoint": label,
        "note": note,
        "n_rollouts": len(lab),
        "acc_first": float(yf.mean()),
        "acc_last": float(y.mean()),
        "n_pos_last": int(y.sum()),
        "mean_blocks": float(lab["n_blocks"].mean()),
        "mean_chars": float(lab["resp_chars"].mean()),
        "template_mean": float(ts.mean()),
        "template_rate_ge3": float((ts >= 3).mean()),
        "template_rate_ge3_wrong": float((ts[y == 0] >= 3).mean()) if (y == 0).any() else np.nan,
        "auroc_template": probes.auroc(y, ts),
        "corr_template_correct": (
            float(np.corrcoef(ts, y)[0, 1]) if ts.std() > 0 and y.std() > 0 else np.nan
        ),
    }
    # Fraction of prompts with at least one correct rollout: the ceiling on how
    # many same-prompt (correct, wrong) pairs Phase 2's patching can build.
    per_prompt = lab.groupby("prompt_idx")["last_block"].agg(["mean", "size"])
    rec["frac_prompts_any_correct"] = float((per_prompt["mean"] > 0).mean())
    rec["frac_prompts_mixed"] = float(
        ((per_prompt["mean"] > 0) & (per_prompt["mean"] < 1)).mean()
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_positives", type=int, default=150,
                    help="rows of the minority class below which a retrained probe is not estimable")
    args = ap.parse_args()

    rows = [r for r in (analyse(l, paths.PAPER_ROOT / p, n) for l, p, n in CHECKPOINTS) if r]
    if not rows:
        raise SystemExit("no eval JSONs found; nothing to scope")
    df = pd.DataFrame(rows).set_index("checkpoint")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Q1: is the collapse in the artefacts we hold? ===")
    print(df[["n_rollouts", "acc_first", "acc_last", "mean_blocks", "mean_chars"]].round(3).to_string())

    print("\n=== Q2: does the H0 confound story hold post-RL? ===")
    print(
        df[["template_mean", "template_rate_ge3", "template_rate_ge3_wrong",
            "auroc_template", "corr_template_correct"]].round(3).to_string()
    )

    print("\n=== Q3: is there enough label variance left to fit a probe? ===")
    q3 = df[["acc_last", "n_pos_last", "frac_prompts_any_correct", "frac_prompts_mixed"]].copy()
    q3["minority_n"] = df.apply(
        lambda r: int(min(r["n_pos_last"], r["n_rollouts"] - r["n_pos_last"])), axis=1
    )
    q3["probe_estimable"] = q3["minority_n"] >= args.min_positives
    print(q3.round(3).to_string())

    print("\n=== verdict ===")
    for name in ("runA_postRL", "runB_postRL"):
        if name not in df.index:
            continue
        r = df.loc[name]
        init = "C_outcome" if "runA" in name else "C_SFT"
        if init not in df.index:
            continue
        base = df.loc[init]
        d_acc = r["acc_first"] - base["acc_first"]
        d_auroc = r["auroc_template"] - base["auroc_template"]
        d_rate = r["template_rate_ge3"] - base["template_rate_ge3"]
        print(
            f"  {name}: accuracy {base['acc_first']:.3f} -> {r['acc_first']:.3f} ({d_acc:+.3f}) | "
            f"template rate {base['template_rate_ge3']:.3f} -> {r['template_rate_ge3']:.3f} ({d_rate:+.3f}) | "
            f"AUROC(template) {base['auroc_template']:.3f} -> {r['auroc_template']:.3f} ({d_auroc:+.3f})"
        )
    print(
        "\n  H0 predicts: accuracy DOWN, template rate UP, AUROC(template) DOWN toward 0.5\n"
        "  (the policy kept the style and dropped the substance).\n"
        "  Reading the other way -- template rate flat or its AUROC preserved --\n"
        "  weakens H0 before any GPU time is spent."
    )
    out = paths.RESULTS / "scope_check.csv"
    metrics_io.write_table(df, out)
    print(f"\n[scope] wrote {out}")


if __name__ == "__main__":
    main()
