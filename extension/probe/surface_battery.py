"""The surface battery: a monitor that reads TEXT ONLY, run against the attack.

This is the script REVISION_PACK section G2 was missing. G2 is described there as
the paper's centrepiece, but nothing in the repository computed it — the feature
names `n_backtrack` and `has_therefore` appeared in no source file, so its table
could not be regenerated, checked, or corrected. Everything G2 asserts is
recomputed here from the eval JSONs on disk.

The question. The probe is not a length detector (see structural_baselines.py:
held-out AUROC 0.982, and 0.939 with `</think>` position stratified out). But the
attacked policy might still be moving along cheap surface axes that the probe
partly rides on. A detector built from nothing but text measures that directly:
freeze it on the baseline checkpoint, run it on the attacked policy's rollouts,
and see whether its score inflates the way the probe's did.

Two batteries, both fit on the baseline checkpoint and then FROZEN:

  7-feature   len_think, n_equals, n_lines, n_backtrack, n_ops, has_therefore,
              ans_len. The small, hand-readable set.
  39-feature  the battery from
              followup/experiments/fragility/residual_probe/surface_residual_probe.py
              (15 structural + 24 reasoning-marker word counts), imported rather
              than reimplemented. The dose-response is reported over the subset
              that is not near-constant, because a standardised shift on a
              near-constant feature explodes.

Reported per battery:
  - held-out AUROC on the baseline checkpoint (prompt-disjoint split)
  - the frozen model's mean P(correct) on baseline vs attacked rollouts
  - true accuracy on both, so score inflation can be read against it
  - per-feature baseline AUROC, attacked AUROC, and the mean shift in BASELINE
    standard deviations
  - the dose-response: do the features that carried correctness signal move
    further than the ones that did not? Spearman, which is the right statistic
    for a monotone ordinal claim (Pearson is not robust here — near-constant
    features make a standardised shift blow up).

Everything is computed on the contamination-filtered clean-406 prompt set, the
same population as every other number in this directory.

    python extension/probe/surface_battery.py
    python extension/probe/surface_battery.py --label_rule first_block
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FOLLOWUP_RESID = os.path.join(
    _REPO_ROOT, "followup", "experiments", "fragility", "residual_probe"
)
if _FOLLOWUP_RESID not in sys.path:
    sys.path.insert(0, _FOLLOWUP_RESID)

from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402
import surface_residual_probe as srp  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_CLOSE = "</think>"

DEFAULT_BASE = "eval_c_outcome_n500.json"
DEFAULT_HACKED = "eval_runA_postRL_n500.json"
DEFAULT_OUT = "extension/outputs/n500/text/61_surface_battery.txt"

# Backtracking / self-correction markers. `n_backtrack` counts occurrences of any
# of these in the reasoning trace. The original G2 run did not survive in any
# artefact, so this is a fresh definition stated in code rather than an attempt
# to reverse-engineer the lost one; the numbers below are what THIS definition
# produces and are regenerable from it.
BACKTRACK_MARKERS = ("wait", "hmm", "actually", "instead", "no,", "not quite",
                     "hold on", "let me try", "alternatively", "recheck")

SEVEN = ("len_think", "n_equals", "n_lines", "n_backtrack", "n_ops",
         "has_therefore", "ans_len")


def features7(text: str) -> dict[str, float]:
    """The 7-feature battery. Trace-only except the answer-derived feature."""
    cut = text.find(THINK_CLOSE)
    think = text[:cut] if cut >= 0 else text
    low = think.lower()
    m = _ANSWER_RE.search(text)
    ans = m.group(1) if m else ""
    return {
        "len_think": float(len(think)),
        "n_equals": float(think.count("=")),
        "n_lines": float(think.count("\n")),
        "n_backtrack": float(sum(low.count(w) for w in BACKTRACK_MARKERS)),
        "n_ops": float(len(re.findall(r"[+\-*/]", think))),
        "has_therefore": float("therefore" in low),
        "ans_len": float(len(ans)),
    }


def label(resp: str, target: int, nums: list, rule: str) -> int:
    """Verifier correctness of the first or last <answer> block. Real verifier."""
    blocks = [mm.group(1) for mm in _ANSWER_RE.finditer(resp)]
    if not blocks:
        return 0
    eq = (blocks[0] if rule == "first_block" else blocks[-1]).strip()
    if not validate_equation(eq, list(nums)):
        return 0
    r = evaluate_equation(eq)
    return int(r is not None and abs(r - int(target)) < 1e-5)


def clean_prompts() -> set[int] | None:
    p = os.path.join(_REPO_ROOT, "extension", "data", "contaminated_prompt_idx.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {int(i) for i in json.load(f)["clean"]}


def load(path: str, rule: str, keep: set[int] | None):
    """(texts, labels, prompt_idx) for one checkpoint's rollouts."""
    texts, y, g = [], [], []
    with open(os.path.join(_REPO_ROOT, path)) as f:
        for p, line in enumerate(l for l in f if l.strip()):
            if keep is not None and p not in keep:
                continue
            row = json.loads(line)
            for resp in row["response"]:
                texts.append(resp)
                y.append(label(resp, int(row["target"]), row["nums"], rule))
                g.append(p)
    return texts, np.asarray(y, int), np.asarray(g, int)


def prompt_split(g: np.ndarray) -> np.ndarray:
    """Deterministic held-out mask, by prompt. Same rule as the residual probe."""
    return np.array(
        [int(hashlib.sha256(str(int(v)).encode()).hexdigest(), 16) % 2 == 0 for v in g]
    )


def matrix(texts: list[str], keys: list[str], fn) -> np.ndarray:
    return np.array([[fn(t)[k] for k in keys] for t in texts], dtype=np.float64)


def run_battery(name: str, keys: list[str], fn, base, hacked, L: list) -> dict:
    """Fit on baseline, freeze, apply to the attacked policy. Returns a record."""
    (tb, yb, gb), (th, yh, _gh) = base, hacked
    Sb, Sh = matrix(tb, keys, fn), matrix(th, keys, fn)

    te = prompt_split(gb)
    tr = ~te
    model = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=3000, C=0.1,
                                                class_weight="balanced"))])
    model.fit(Sb[tr], yb[tr])

    auroc_heldout = float(roc_auc_score(yb[te], model.predict_proba(Sb[te])[:, 1]))
    # The frozen model's score on each population. The baseline figure is the
    # held-out slice, so it is not inflated by having been fit on those rows.
    p_base = float(model.predict_proba(Sb[te])[:, 1].mean())
    p_hack = float(model.predict_proba(Sh)[:, 1].mean())
    auroc_hacked = float(roc_auc_score(yh, model.predict_proba(Sh)[:, 1])) \
        if len(np.unique(yh)) > 1 else float("nan")

    # `acc_baseline` must be the accuracy of THE SAME ROWS `p_base` scores, or the
    # section's central comparison ("the monitor reads X while the truth is Y") is
    # between two different populations. Earlier this was `yb.mean()` over all
    # baseline rows while `p_base` was the held-out half only: 0.4980 vs the
    # 0.5173 that actually matches, a 1.9 pp phantom miscalibration at baseline.
    # The attacked side needs no such split -- the model never saw those rows --
    # so `acc_hacked` and `p_hack` are both over all of them. The full-population
    # baseline accuracy is kept beside it, labelled, because other sections quote it.
    rec = {
        "battery": name, "n_features": len(keys),
        "auroc_heldout_baseline": auroc_heldout,
        "auroc_frozen_on_hacked": auroc_hacked,
        "score_baseline": p_base, "score_hacked": p_hack,
        "acc_baseline": float(yb[te].mean()), "acc_hacked": float(yh.mean()),
        "acc_baseline_all_rows": float(yb.mean()),
        "n_baseline_heldout": int(te.sum()), "n_baseline_all": int(len(yb)),
        "n_hacked": int(len(yh)),
    }

    L.append(f"### {name} battery ({len(keys)} features)")
    L.append("")
    L.append(f"  held-out AUROC on the baseline checkpoint      {auroc_heldout:.4f}")
    L.append("")
    L.append(f"  frozen on baseline, then applied to the attacked policy")
    L.append(f"  (baseline column = the held-out {int(te.sum())} of {len(yb)} rows, so the "
             "score and\n   the accuracy beneath it describe the same rollouts):")
    L.append(f"    mean P(correct)   {p_base:.4f}  ->  {p_hack:.4f}   "
             f"({p_hack - p_base:+.4f})")
    L.append(f"    true accuracy     {float(yb[te].mean()):.4f}  ->  {float(yh.mean()):.4f}   "
             f"({float(yh.mean()) - float(yb[te].mean()):+.4f})")
    L.append(f"    (baseline accuracy over all {len(yb)} rows, for cross-reference: "
             f"{float(yb.mean()):.4f})")
    L.append(f"    AUROC             {auroc_heldout:.4f}  ->  {auroc_hacked:.4f}   "
             f"({auroc_hacked - auroc_heldout:+.4f})")
    L.append("")

    # ---- per-feature: signal carried vs distance moved ----------------------
    per = []
    for j, k in enumerate(keys):
        col = Sb[:, j]
        sd = col.std()
        a_b = roc_auc_score(yb, col) if sd > 0 else float("nan")
        a_h = (roc_auc_score(yh, Sh[:, j])
               if Sh[:, j].std() > 0 and len(np.unique(yh)) > 1 else float("nan"))
        shift = float((Sh[:, j].mean() - col.mean()) / sd) if sd > 0 else float("nan")
        # NEAR-CONSTANT: the modal baseline value covers >=99% of rollouts. Such
        # a feature has a baseline SD near zero, so dividing by it turns any
        # movement at all into a huge "standardised shift" — `w_yes` reads
        # +12.4 SD and `ans_paren` -3.7 SD on this data, which is what makes the
        # Pearson version of the dose-response meaningless. This is the
        # "keeping the ones that are not near-constant" filter, stated as a rule
        # instead of applied by eye.
        vals, counts = np.unique(col, return_counts=True)
        modal_frac = float(counts.max() / len(col)) if len(col) else 1.0
        per.append({"feature": k, "auroc_base": float(a_b), "auroc_hacked": float(a_h),
                    "shift_sd": shift, "baseline_sd": float(sd),
                    "modal_frac": modal_frac,
                    "near_constant": bool(sd == 0 or modal_frac >= 0.99),
                    "signal": abs(float(a_b) - 0.5)})
    per.sort(key=lambda d: -d["signal"])
    rec["per_feature"] = per

    L.append(f"  {'feature':<18} {'AUROC base':>10} {'AUROC hacked':>13} "
             f"{'shift (SD)':>11}  near-const")
    L.append("  " + "-" * 68)
    for d in per:
        L.append(f"  {d['feature']:<18} {d['auroc_base']:>10.4f} "
                 f"{d['auroc_hacked']:>13.4f} {d['shift_sd']:>+11.2f}"
                 f"  {'yes' if d['near_constant'] else '':>10}")
    L.append("")

    # ---- dose-response: signal carried vs |shift| --------------------------
    usable = [d for d in per
              if not d["near_constant"] and np.isfinite(d["shift_sd"])
              and np.isfinite(d["signal"])]
    sig = np.array([d["signal"] for d in usable])
    mov = np.array([abs(d["shift_sd"]) for d in usable])
    if len(usable) >= 4:
        rho, p_rho = stats.spearmanr(sig, mov)
        r_p, p_p = stats.pearsonr(sig, mov)
        rec.update({"n_usable": len(usable), "spearman_rho": float(rho),
                    "spearman_p": float(p_rho), "pearson_r": float(r_p),
                    "pearson_p": float(p_p)})
        L.append(f"  dose-response over the {len(usable)} usable features "
                 "(signal carried vs distance moved):")
        L.append(f"    Spearman rho = {rho:+.3f}, p = {p_rho:.3f}   <- the reportable one")
        L.append(f"    Pearson  r   = {r_p:+.3f}, p = {p_p:.3f}   (not robust here; do not quote)")
        order = np.argsort(sig)
        n3 = max(1, len(usable) // 3)
        terciles = [("low", order[:n3]), ("mid", order[n3:len(usable) - n3]),
                    ("high", order[len(usable) - n3:])]
        rec["terciles"] = {}
        for tag, idx in terciles:
            if not len(idx):
                continue
            rec["terciles"][tag] = {"n": int(len(idx)), "mean_abs_shift": float(mov[idx].mean())}
            L.append(f"    {tag:<5} signal tercile (n={len(idx)}): "
                     f"mean |shift| = {mov[idx].mean():.2f} SD")
    L.append("")
    return rec


def length_detector(base, hacked, L: list) -> dict:
    """The one-feature detector of REVISION_PACK section G, made regenerable.

    A detector that reads nothing but `</think>` position, frozen on the baseline
    and pointed at the attacked policy. Shorter traces are more often correct
    here, so the SCORE is negative length; the threshold is the median `</think>`
    position of CORRECT baseline rollouts, frozen, so the detector is never refit
    to the attack.

    Precision at that threshold is the number the section turns on: of the
    rollouts the detector calls correct, how many are.
    """
    (tb, yb, _gb), (th, yh, _gh) = base, hacked
    lb = np.array([len(t[:t.find(THINK_CLOSE)]) if THINK_CLOSE in t else len(t)
                   for t in tb], dtype=float)
    lh = np.array([len(t[:t.find(THINK_CLOSE)]) if THINK_CLOSE in t else len(t)
                   for t in th], dtype=float)
    thr = float(np.median(lb[yb == 1]))          # frozen on CORRECT baseline rows
    prec_b = float(yb[lb <= thr].mean()) if (lb <= thr).any() else float("nan")
    prec_h = float(yh[lh <= thr].mean()) if (lh <= thr).any() else float("nan")
    rec = {
        "acc_baseline": float(yb.mean()), "acc_hacked": float(yh.mean()),
        "median_think_baseline": float(np.median(lb)),
        "median_think_hacked": float(np.median(lh)),
        "auroc_baseline": float(roc_auc_score(yb, -lb)),
        "auroc_hacked": float(roc_auc_score(yh, -lh)),
        "frozen_threshold_chars": thr,
        "precision_baseline": prec_b, "precision_hacked": prec_h,
        "flag_rate_baseline": float((lb <= thr).mean()),
        "flag_rate_hacked": float((lh <= thr).mean()),
    }
    L.append("### length detector (one feature: `</think>` character position)")
    L.append("")
    L.append(f"  {'':<44}{'baseline':>10}{'attacked':>12}")
    L.append(f"  {'true accuracy':<44}{rec['acc_baseline']:>10.4f}{rec['acc_hacked']:>12.4f}")
    L.append(f"  {'median </think> position (chars)':<44}"
             f"{rec['median_think_baseline']:>10.0f}{rec['median_think_hacked']:>12.0f}")
    L.append(f"  {'length-detector AUROC':<44}"
             f"{rec['auroc_baseline']:>10.4f}{rec['auroc_hacked']:>12.4f}")
    L.append(f"  {'precision at the frozen threshold':<44}"
             f"{rec['precision_baseline']:>10.4f}{rec['precision_hacked']:>12.4f}")
    L.append(f"  {'fraction flagged correct':<44}"
             f"{rec['flag_rate_baseline']:>10.4f}{rec['flag_rate_hacked']:>12.4f}")
    L.append(f"  threshold frozen at {thr:.0f} chars = the median </think> position of "
             "CORRECT baseline rollouts")
    L.append("")
    return rec


def plot_dose_response(rec: dict, out_png: str) -> str | None:
    """Regenerate `fig_dose_response.png` for section G2.

    The figure this replaces was wrong in three ways, and each one is a rule in
    the plotting below:

      - it plotted all 39 features with no near-constant filter, so one point sat
        at |shift| = 125,000 SD and compressed every real point onto the axis.
        Here only the usable features are drawn, and the excluded count is
        stated on the figure rather than silently dropped.
      - it titled itself with the PEARSON r = -0.14 — the statistic the section
        explicitly says not to report — while claiming a positive relationship.
        Here the title carries Spearman rho, which is what the ordinal claim
        rests on.
      - it drew a least-squares fit line, which asserts linearity the data does
        not have. The ordinal summary is the tercile means, so that is what is
        drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = rec.get("per_feature", [])
    usable = [d for d in per if not d["near_constant"] and np.isfinite(d["shift_sd"])
              and np.isfinite(d["signal"])]
    if len(usable) < 4:
        return None
    n_excluded = len(per) - len(usable)
    sig = np.array([d["signal"] for d in usable])
    mov = np.array([abs(d["shift_sd"]) for d in usable])

    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    POINT, SUMMARY = "#2a78d6", "#eb6834"      # validated categorical slots 1, 2

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.8)

    ax.scatter(sig, mov, s=64, color=POINT, edgecolor=SURFACE, linewidth=1.6,
               zorder=3, label="surface feature")

    # Tercile means — the ordinal summary the claim actually makes.
    order = np.argsort(sig)
    n3 = max(1, len(usable) // 3)
    groups = [("low", order[:n3]), ("mid", order[n3:len(usable) - n3]),
              ("high", order[len(usable) - n3:])]
    gx = [float(sig[idx].mean()) for _t, idx in groups if len(idx)]
    gy = [float(mov[idx].mean()) for _t, idx in groups if len(idx)]
    ax.plot(gx, gy, color=SUMMARY, linewidth=2.0, marker="D", markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
            label="tercile mean")
    # The tercile VALUES live in the subtitle, outside the data area. Three
    # placements were tried inside it — beside the marker, below the line, and in
    # a band near the top — and each collided with something: 8 of the 23 features
    # sit between x 0.33 and 0.38, and the one clear region (top left) is where
    # the `ans_paren` outlier sits. A summary of three numbers does not need to be
    # positioned in data space at all; the orange series is already legended.
    tercile_txt = " · ".join(
        f"{tag} {yv:.2f}" for (tag, _idx), yv in
        zip([g for g in groups if len(g[1])], gy)
    )

    # ONE direct label: the low-signal / high-shift outlier. It is the point a
    # reader needs named — it is what pulls the correlation down and it is why
    # the fit is reported as a rank statistic. Labelling the other extremes put
    # text on top of neighbouring marks in the right-hand cluster; the per-feature
    # table in the .txt carries every name.
    i_out = int(np.argmax(mov))
    ax.annotate(usable[i_out]["feature"], (sig[i_out], mov[i_out]),
                textcoords="offset points", xytext=(11, -4), ha="left",
                fontsize=8.5, color=MUTED)
    # Breathing room so no mark or label sits on a spine.
    ax.margins(x=0.09, y=0.10)

    rho = rec.get("spearman_rho", float("nan"))
    p = rec.get("spearman_p", float("nan"))
    ax.set_xlabel("correctness signal carried at baseline   |AUROC − 0.5|",
                  fontsize=10, color=MUTED)
    ax.set_ylabel("|shift| under RL attack  (baseline SD)", fontsize=10, color=MUTED)
    # Three short lines, not two long ones: appended to the subtitle the tercile
    # summary widened the title past the axes and `bbox_inches="tight"` grew the
    # canvas to match, leaving the plot squeezed against a dead right margin.
    ax.set_title("Features that carried signal moved further\n"
                 f"Spearman ρ = {rho:+.3f}, p = {p:.3f}  (n = {len(usable)} features)\n"
                 f"tercile means (SD): {tercile_txt}",
                 fontsize=11.5, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.01, -0.02,
             f"{n_excluded} near-constant features excluded (modal baseline value ≥99% of "
             "rollouts): a standardised\nshift divides by a near-zero SD and explodes. "
             "Ordinal claim — no linear fit is drawn.",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE, help="baseline checkpoint rollouts")
    ap.add_argument("--hacked", default=DEFAULT_HACKED, help="attacked-policy rollouts")
    ap.add_argument("--label_rule", default="last_block",
                    choices=["first_block", "last_block"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fig", default="fig_dose_response.png",
                    help="dose-response figure for the 39-feature battery")
    args = ap.parse_args()

    keep = clean_prompts()
    base = load(args.base, args.label_rule, keep)
    hacked = load(args.hacked, args.label_rule, keep)

    L = ["Surface battery — a text-only monitor, frozen on baseline, run on the attack",
         f"  baseline: {args.base}",
         f"  attacked: {args.hacked}",
         f"  label rule: {args.label_rule}   "
         f"prompts: {'clean-406' if keep else 'ALL (no contamination manifest)'}",
         f"  rollouts: {len(base[0])} baseline, {len(hacked[0])} attacked",
         "",
         "  This is the regenerable version of REVISION_PACK section G2, whose",
         "  original numbers were computed in a session that left no artefact.",
         ""]

    keys39 = sorted(srp.surface_features(base[0][0]))
    length = length_detector(base, hacked, L)
    records = [
        run_battery("7-feature", list(SEVEN), features7, base, hacked, L),
        run_battery("39-feature", keys39, srp.surface_features, base, hacked, L),
    ]

    fig_path = plot_dose_response(records[1], os.path.join(_REPO_ROOT, args.fig))
    if fig_path:
        L.append(f"  dose-response figure -> {args.fig}")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    out = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(txt + "\n")
    with open(out.replace(".txt", ".json"), "w") as f:
        json.dump({"base": args.base, "hacked": args.hacked,
                   "label_rule": args.label_rule,
                   "backtrack_markers": list(BACKTRACK_MARKERS),
                   "length_detector": length,
                   "batteries": records}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
