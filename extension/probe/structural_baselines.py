"""Structural controls for the probe: how much of it is just "how long did I think?"

The probe reads the residual stream AT the `</think>` token. Where that token
falls in the sequence is itself a strong correctness signal for this policy,
because it rambles when it is wrong. So before crediting the probe with reading
correctness, we have to ask what a model with access to nothing but that
position would score.

This script answers three questions, all on the same population and the same
GroupKFold(5)-by-prompt folds the probe uses:

  (1) DISCRIMINATION. Held-out balanced AUROC of
        - the 896-d L16 activation (the probe), vs
        - one scalar: the token index of `</think>`, vs
        - a small structural feature vector (position, char length, #blocks).
      Plus the probe's AUROC STRATIFIED within `</think>`-position deciles,
      which is the probe's discrimination at (approximately) matched length.

  (2) SELECTION. best-of-K accuracy for
        - random pick (the expected accuracy of ONE of the K, which is the right
          denominator for a best-of-K lift -- NOT first-rollout pass@1),
        - "pick the shortest <think> body" -- no probe, no forward pass,
        - the 39-feature surface model -- text only, no forward pass,
        - probe-best-of-K,
        - oracle pass@K.
      This is the honest denominator for the applied-probe lift. The effective K
      is reported too: uncached rollouts mean K is a ceiling, not a constant.

  (3) POPULATION. Rollouts with no locatable `</think>` are absent from the
      cache entirely. This reports how many there are and how accurate they are,
      because they are excluded from every AUROC in the paper but are very much
      present at deployment.

Both outcomes are publishable. If the probe collapses to the structural baseline
under stratification, the "near-oracle internal verifier" framing is wrong. If it
survives, the paper gets a control it currently lacks. Do not tune anything here
to make one come out.

Pure CPU, reads the local caches + eval JSONs.

    python extension/probe/structural_baselines.py
    python extension/probe/structural_baselines.py --layer 16 --ckpt C_outcome
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_CLOSE = "</think>"

DEFAULT_CACHE = "extension/cache/probe_cache_n500_clean406"
DEFAULT_EVALS = {"C_SFT": "eval_c_sft_n500.json", "C_outcome": "eval_c_outcome_n500.json"}
DEFAULT_OUT = "extension/outputs/n500/text/60_structural_baselines.txt"


def check_block(eq: str, target: int, nums: list) -> bool:
    """Verifier semantics for one <answer> block — the REAL verifier.

    This used to be a local regex + `eval` reimplementation. It agreed with
    `evaluation.countdown` on all 22,995 answer-bearing rollouts of
    eval_c_{outcome,sft}_n500 and eval_runA_postRL_n500, so nothing in this
    script's published numbers moves — but a second implementation of the label
    rule is exactly what the rest of the project refuses to keep (see
    followup/fragility_core/labels.py: labels must be bit-identical to the RL
    reward's notion of correctness, and two of this project's historical bugs
    came from a second implementation drifting from the first).
    """
    eq = eq.strip()
    if not validate_equation(eq, list(nums)):
        return False
    result = evaluate_equation(eq)
    return result is not None and abs(result - int(target)) < 1e-5


def rollout_features(eval_path: str, keep_prompts: set | None = None,
                     want_text: bool = False) -> tuple[dict, dict]:
    """(prompt_idx, resp_idx) -> dict of label + purely structural features.

    `keep_prompts` restricts to a prompt subset. It has to be passed, because the
    eval JSON holds all 500 prompts while every cache in this script is
    contamination-filtered to the clean 406. Sections (3) and (4) below compare
    cached against uncached rollouts, so if the two sides come from different
    prompt sets the 94 contaminated prompts (1,504 rollouts) read as "the probe
    could not score these" — which is how section (4) briefly reported a
    full-population AUROC of 0.753 against a cached-subset 0.982.
    """
    out, texts = {}, {}
    for p, row in enumerate(json.loads(l) for l in open(eval_path) if l.strip()):
        if keep_prompts is not None and p not in keep_prompts:
            continue
        t = int(row["target"]); nums = list(row["nums"])
        for r_i, resp in enumerate(row["response"]):
            m = _ANSWER_RE.search(resp)
            close = resp.find(THINK_CLOSE)
            if want_text:
                texts[(p, r_i)] = resp
            out[(p, r_i)] = {
                "label": int(bool(m) and check_block(m.group(1), t, nums)),
                "has_think_close": close >= 0,
                "think_chars": close if close >= 0 else len(resp),
                "resp_chars": len(resp),
                "n_blocks": len(_ANSWER_RE.findall(resp)),
            }
    return out, texts


def _pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.1, max_iter=2000))])


def _surface_matrix(texts: list) -> np.ndarray:
    """The 39-feature surface battery, IMPORTED from the module that defines it.

    Not reimplemented: `surface_residual_probe.surface_features` is the artefact
    the residualisation numbers were measured with, and a second copy of the
    feature list here would drift from it silently.
    """
    resid_dir = os.path.join(
        _REPO_ROOT, "followup", "experiments", "fragility", "residual_probe")
    if resid_dir not in sys.path:
        sys.path.insert(0, resid_dir)
    import surface_residual_probe as srp

    keys = sorted(srp.surface_features(texts[0]))
    return srp.feature_matrix(texts, keys)


def prompt_bootstrap(per_prompt: dict[str, list], *, n_boot: int = 10000,
                     seed: int = 0, ci: float = 0.95) -> dict:
    """Prompt-clustered paired bootstrap over the selector arms.

    `per_prompt` maps arm name -> a per-prompt outcome list, all in the SAME
    prompt order. One resample of prompts is drawn and EVERY arm is read off it,
    so a prompt appears on both sides of every contrast in the same draw — that
    pairing is what makes probe-minus-shortest tight even though the two arms'
    absolute accuracies are noisy.

    The prompt is the resampling unit because the 16 rollouts of one problem
    share its difficulty; resampling rollouts would understate the SE.

    REVISION_PACK B1's previous intervals were computed against a point estimate
    that has since been withdrawn, so they are recomputed here rather than
    rescaled, and they live in the artefact instead of in a markdown table.
    """
    names = list(per_prompt)
    M = np.array([per_prompt[k] for k in names], dtype=float)   # (arms, prompts)
    n = M.shape[1]
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0
    draws = np.empty((len(names), n_boot))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[:, b] = M[:, idx].mean(axis=1)
    out = {"n_prompts": n, "n_boot": n_boot}
    for i, k in enumerate(names):
        lo, hi = np.quantile(draws[i], [lo_q, 1 - lo_q])
        out[k] = {"est": float(M[i].mean()), "ci_lo": float(lo), "ci_hi": float(hi)}
    return out, names, draws, M


def contrast(names, draws, M, a: str, b: str, *, ci: float = 0.95) -> dict:
    """Paired difference a - b with a percentile CI from the shared draws."""
    ia, ib = names.index(a), names.index(b)
    d = draws[ia] - draws[ib]
    lo_q = (1.0 - ci) / 2.0
    lo, hi = np.quantile(d, [lo_q, 1 - lo_q])
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return {"est": float(M[ia].mean() - M[ib].mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(1.0, max(p, 1.0 / len(d)))),
            "significant": bool(lo > 0 or hi < 0)}


def heldout_scores(F: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(F, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        s[te] = _pipe().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    return s


def balanced_auroc(y: np.ndarray, s: np.ndarray, seed: int = 0) -> float:
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    if nb < 5:
        return float("nan")
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(pos, nb, replace=False),
                          rng.choice(neg, nb, replace=False)])
    return float(roc_auc_score(y[idx], s[idx]))


def stratified_auroc(y: np.ndarray, s: np.ndarray, strat: np.ndarray,
                     n_bins: int = 10) -> float:
    """AUROC computed WITHIN bins of `strat`, pooled by discordant-pair count.

    This is the probe's discrimination holding the stratifier ~constant. A
    feature that is a pure function of `strat` scores ~0.5 here by construction,
    which is the sanity check reported alongside.
    """
    m = ~np.isnan(s)
    y, s, strat = y[m], s[m], strat[m]
    edges = np.quantile(strat, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.digitize(strat, edges[1:-1]), 0, n_bins - 1)
    num = den = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.sum() < 2 or len(np.unique(y[sel])) < 2:
            continue
        n1 = int(y[sel].sum()); n0 = int((1 - y[sel]).sum())
        num += roc_auc_score(y[sel], s[sel]) * n1 * n0
        den += n1 * n0
    return float(num / den) if den else float("nan")


def analyse(ckpt: str, eval_path: str, cache_dir: str, layer: int, K: int, L: list,
            *, with_surface: bool = True, n_boot: int = 10000) -> dict:
    npz = os.path.join(cache_dir, f"{ckpt}_l{layer}_pre_answer.npz")
    if not os.path.exists(npz):
        L.append(f"  [skip] {npz} not found")
        return {}
    with np.load(npz) as d:
        X = d["X"]
    meta = json.load(open(npz.replace(".npz", ".meta.json")))
    # The cache's OWN prompt set is the population for every section here. Taken
    # from the cache rather than from the contamination manifest so the two can
    # never disagree.
    cache_prompts = {int(m["prompt_idx"]) for m in meta}
    feats, _texts = rollout_features(eval_path, keep_prompts=cache_prompts,
                                     want_text=with_surface)

    g = np.array([int(m["prompt_idx"]) for m in meta])
    keys = [(int(m["prompt_idx"]), int(m["resp_idx"])) for m in meta]
    y = np.array([feats[k]["label"] for k in keys], dtype=int)
    tok_idx = np.array([int(m["tok_idx"]) for m in meta], dtype=float)
    think_chars = np.array([feats[k]["think_chars"] for k in keys], dtype=float)
    n_blocks = np.array([feats[k]["n_blocks"] for k in keys], dtype=float)
    # Rollout text in cache-row order, for the surface battery below.
    surface_texts = ([_texts[k] for k in keys] if with_surface else None)

    # ---- (1) discrimination -------------------------------------------------
    s_probe = heldout_scores(X, y, g)
    s_pos = heldout_scores(tok_idx.reshape(-1, 1), y, g)
    s_struct = heldout_scores(
        np.column_stack([tok_idx, think_chars, n_blocks]), y, g)

    # The 39-feature surface battery, as a THIRD free baseline. It belongs in
    # this artefact rather than in a loose session: REVISION_PACK B2 quoted it
    # at best-of-16 0.6429 / AUROC 0.9252, but 0.9252 is the shipped
    # surface-only probe's number under a different label rule and a different
    # split, sat next to an in-sample probe AUROC. Computed here it is on this
    # script's population, this script's label rule, and this script's folds, so
    # all four selector rows below are finally comparable.
    s_surface = None
    if surface_texts is not None:
        try:
            S = _surface_matrix(surface_texts)
            s_surface = heldout_scores(S, y, g)
        except Exception as exc:  # the followup module may not be importable
            L.append(f"  [note] 39-feature surface baseline skipped: {exc}")

    # In-sample counterpart of the two probe rows. Reported ONLY so the gap is
    # on the record: a probe fit and scored on the same 6,306 rows reads 0.995
    # unstratified and 0.981 stratified, against 0.982 / 0.939 held out, and
    # those in-sample figures were quoted in REVISION_PACK B2 next to a scalar
    # baseline that needs no fitting. Never quote the in-sample column.
    s_probe_insample = _pipe().fit(X, y).predict_proba(X)[:, 1]

    res = {
        "ckpt": ckpt, "layer": layer, "n_rows": int(len(y)),
        "n_prompts": int(len(set(g.tolist()))), "pos_frac": float(y.mean()),
        "auroc_probe": balanced_auroc(y, s_probe),
        "auroc_thinkclose_position": balanced_auroc(y, s_pos),
        "auroc_structural_3feat": balanced_auroc(y, s_struct),
        "auroc_probe_stratified_by_position": stratified_auroc(y, s_probe, tok_idx),
        "auroc_position_stratified_by_position": stratified_auroc(y, s_pos, tok_idx),
        "auroc_probe_INSAMPLE": balanced_auroc(y, s_probe_insample),
        "auroc_probe_stratified_INSAMPLE": stratified_auroc(y, s_probe_insample, tok_idx),
    }
    if s_surface is not None:
        res["auroc_surface39"] = balanced_auroc(y, s_surface)
        res["auroc_surface39_stratified_by_position"] = stratified_auroc(y, s_surface, tok_idx)

    L.append(f"### {ckpt}  L{layer}   n_rows={res['n_rows']}  "
             f"n_prompts={res['n_prompts']}  pos_frac={res['pos_frac']:.3f}")
    L.append("  label rule: FIRST <answer> block (the paper's probe-AUROC rule), "
             "verifier = evaluation.countdown")
    L.append("")
    L.append("  (1) DISCRIMINATION — held-out balanced AUROC, GroupKFold(5) by prompt")
    L.append(f"      probe (L{layer} residual, {X.shape[1]}-d)          {res['auroc_probe']:.3f}")
    L.append(f"      ONE SCALAR: </think> token index               {res['auroc_thinkclose_position']:.3f}")
    L.append(f"      structural 3-feature (pos, chars, #blocks)     {res['auroc_structural_3feat']:.3f}")
    if s_surface is not None:
        L.append(f"      39-feature surface battery (text only)        {res['auroc_surface39']:.3f}")
    L.append(f"      probe, STRATIFIED by </think>-position decile  {res['auroc_probe_stratified_by_position']:.3f}")
    L.append(f"      position, stratified the same way (sanity)     {res['auroc_position_stratified_by_position']:.3f}")
    L.append("")
    L.append("      [IN-SAMPLE, do not quote — fit and scored on the same rows]")
    L.append(f"      probe, in-sample                              {res['auroc_probe_INSAMPLE']:.3f}")
    L.append(f"      probe, in-sample + stratified                 {res['auroc_probe_stratified_INSAMPLE']:.3f}")
    L.append("      The held-out rows above are the reportable ones. The stratification")
    L.append("      cost is what differs most: held out it is "
             f"{res['auroc_probe'] - res['auroc_probe_stratified_by_position']:+.3f}, in sample only "
             f"{res['auroc_probe_INSAMPLE'] - res['auroc_probe_stratified_INSAMPLE']:+.3f}.")
    L.append("")

    # ---- (2) selection ------------------------------------------------------
    by_p_cached: dict[int, list] = defaultdict(list)
    for i, k in enumerate(keys):
        if np.isnan(s_probe[i]):
            continue
        by_p_cached[k[0]].append((k[1], float(s_probe[i]), int(y[i]), float(think_chars[i]),
                                  float(s_surface[i]) if s_surface is not None
                                  and not np.isnan(s_surface[i]) else float("-inf")))
    for p in by_p_cached:
        by_p_cached[p].sort()

    prompts = sorted(by_p_cached)
    rand_pick, short_pick, probe_pick, oracle, surf_pick = [], [], [], [], []
    for p in prompts:
        v = by_p_cached[p][:K]
        if not v:
            continue
        labs = np.array([t[2] for t in v])
        rand_pick.append(labs.mean())
        short_pick.append(v[int(np.argmin([t[3] for t in v]))][2])
        probe_pick.append(v[int(np.argmax([t[1] for t in v]))][2])
        oracle.append(float(labs.max() == 1))
        if s_surface is not None:
            surf_pick.append(v[int(np.argmax([t[4] for t in v]))][2])
    rp, sp, pp, orc = map(float, map(np.mean, (rand_pick, short_pick, probe_pick, oracle)))
    gap = max(1e-9, orc - rp)
    # The EFFECTIVE K. Rollouts with no cached probe score are absent, so a
    # prompt whose model rambled offers fewer than K candidates and "best-of-K"
    # is really best-of-min(K, n_cached). Every arm here (including `random` and
    # `oracle`) is computed on the same reduced candidate set, so the contrast
    # is internally consistent — but the K in the label is a ceiling, not a
    # constant, and a reader comparing against an uncached best-of-16 must know.
    k_eff = [len(by_p_cached[p][:K]) for p in prompts if by_p_cached[p]]
    res.update({
        "select_random": rp, "select_shortest_think": sp,
        "select_probe": pp, "select_oracle": orc,
        "lift_shortest_pp": 100 * (sp - rp), "lift_probe_pp": 100 * (pp - rp),
        "probe_over_shortest_pp": 100 * (pp - sp),
        "select_mean_effective_K": float(np.mean(k_eff)) if k_eff else float("nan"),
        "select_min_effective_K": int(np.min(k_eff)) if k_eff else 0,
        "select_frac_prompts_full_K": float(np.mean([k == K for k in k_eff])) if k_eff else float("nan"),
    })
    L.append(f"  (2) SELECTION — best-of-{K} on prompts with cached rollouts (n={len(rand_pick)})")
    L.append(f"      effective K: mean {res['select_mean_effective_K']:.2f}, min "
             f"{res['select_min_effective_K']}, "
             f"{100*res['select_frac_prompts_full_K']:.1f}% of prompts have the full {K}")
    L.append(f"      random pick (expected acc of 1 of {K})  {rp:.4f}")
    L.append(f"      pick SHORTEST <think> (no probe)      {sp:.4f}   "
             f"{100*(sp-rp):+.1f} pp   ({100*(sp-rp)/gap:.0f}% of oracle gap)")
    if surf_pick:
        fp = float(np.mean(surf_pick))
        res["select_surface39"] = fp
        res["lift_surface39_pp"] = 100 * (fp - rp)
        res["probe_over_surface39_pp"] = 100 * (pp - fp)
        L.append(f"      39-feature surface model (no probe)   {fp:.4f}   "
                 f"{100*(fp-rp):+.1f} pp   ({100*(fp-rp)/gap:.0f}% of oracle gap)")
    L.append(f"      probe best-of-{K:<2}                     {pp:.4f}   "
             f"{100*(pp-rp):+.1f} pp   ({100*(pp-rp)/gap:.0f}% of oracle gap)")
    L.append(f"      oracle pass@{K:<2}                       {orc:.4f}")
    L.append(f"      >>> probe's gain OVER the free structural baseline: "
             f"{100*(pp-sp):+.1f} pp")
    if surf_pick:
        L.append(f"      >>> probe's gain OVER the 39-feature surface model:  "
                 f"{100*(pp-float(np.mean(surf_pick))):+.1f} pp")
    L.append("      NOTE: 'random pick' is the expected accuracy of ONE of the K cached")
    L.append("      rollouts, which is the right denominator for a best-of-K lift. It is")
    L.append("      NOT the same estimator as first-rollout pass@1, and it is not the")
    L.append("      pass@1 in probe_usefulness_suite_results_n406.json (a different")
    L.append("      rollout sample). Do not mix the two inside one table.")
    L.append("")

    # ---- (2b) prompt-clustered paired bootstrap ------------------------------
    arms = {"random": rand_pick, "shortest": short_pick,
            "probe": probe_pick, "oracle": oracle}
    if surf_pick:
        arms["surface39"] = surf_pick
    boot, names, draws, M = prompt_bootstrap(arms, n_boot=n_boot, seed=0)
    res["bootstrap"] = {"n_prompts": boot["n_prompts"], "n_boot": boot["n_boot"],
                        "arms": {k: boot[k] for k in names}}
    L.append(f"  (2b) PROMPT-CLUSTERED PAIRED BOOTSTRAP ({n_boot} resamples, "
             f"n={boot['n_prompts']} prompts)")
    L.append(f"      {'arm':<12}{'accuracy':>10}   95% CI")
    for k in names:
        b = boot[k]
        L.append(f"      {k:<12}{b['est']:>10.4f}   [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]")
    L.append("")
    L.append(f"      {'contrast':<24}{'delta pp':>10}   95% CI (pp)        p")
    pairs = [("shortest", "random"), ("probe", "random"), ("probe", "shortest"),
             ("oracle", "random")]
    if surf_pick:
        pairs.insert(3, ("probe", "surface39"))
        pairs.insert(3, ("surface39", "random"))
    res["contrasts"] = {}
    for a, b in pairs:
        c = contrast(names, draws, M, a, b)
        res["contrasts"][f"{a}_minus_{b}"] = c
        L.append(f"      {a + ' - ' + b:<24}{100*c['est']:>+10.2f}   "
                 f"[{100*c['ci_lo']:+.2f}, {100*c['ci_hi']:+.2f}]"
                 f"{'':<4}{c['p']:.4f}{'  *' if c['significant'] else ''}")
    # Share of the oracle headroom, with a CI. The ratio is bootstrapped as a
    # ratio -- taking (probe-random)/(oracle-random) from the point estimates and
    # attaching the numerator's CI would understate it, because the denominator
    # is estimated on the same 406 prompts and moves with it.
    ir, ip, io = names.index("random"), names.index("probe"), names.index("oracle")
    den = draws[io] - draws[ir]
    ok = den > 0
    share = (draws[ip] - draws[ir])[ok] / den[ok]
    lo, hi = np.quantile(share, [0.025, 0.975])
    res["probe_share_of_headroom"] = {"est": float((pp - rp) / gap),
                                      "ci_lo": float(lo), "ci_hi": float(hi)}
    L.append("")
    L.append(f"      probe's share of oracle headroom  {100*(pp-rp)/gap:.1f}%   "
             f"[{100*lo:.1f}%, {100*hi:.1f}%]")
    L.append("      (bootstrapped as a ratio: the denominator is estimated on the same")
    L.append("      prompts as the numerator and moves with it.)")
    L.append("")

    # ---- (3) population -----------------------------------------------------
    have, miss = [], []
    for k, f in feats.items():
        (have if f["has_think_close"] else miss).append(f["label"])
    res.update({
        "n_prompts_population": len(cache_prompts),
        "n_with_think_close": len(have), "n_without_think_close": len(miss),
        "acc_with_think_close": float(np.mean(have)) if have else float("nan"),
        "acc_without_think_close": float(np.mean(miss)) if miss else float("nan"),
    })
    L.append(f"  (3) POPULATION — the cache's own {len(cache_prompts)} prompts "
             f"({len(feats)} rollouts); no-</think> rollouts are excluded from the cache")
    L.append(f"      with </think>:    n={len(have):>5}  accuracy={res['acc_with_think_close']:.4f}")
    L.append(f"      WITHOUT </think>: n={len(miss):>5}  accuracy={res['acc_without_think_close']:.4f}"
             f"   ({100*len(miss)/max(1,len(have)+len(miss)):.1f}% of all rollouts, dropped from every AUROC)")
    L.append("")

    # ---- (4) DEPLOYMENT AUROC ----------------------------------------------
    # Every AUROC above (and in the rest of the paper) is computed on the cached
    # subset only, i.e. after silently discarding the rollouts that never emitted
    # `</think>`. At deployment you do not get to discard those -- you get to
    # SCORE them, and the natural rule is "no </think> => predict wrong", which on
    # this task is right ~100% of the time. That rule is free and makes the
    # monitor strictly better, so reporting only the filtered AUROC understates
    # the deployable system while overstating what the linear probe itself does.
    # Here: pool the cached rollouts' probe scores with a score of -inf (encoded
    # as one below the minimum) for every uncached rollout, then one AUROC over
    # the FULL population.
    cached_keys = set(keys)
    full_y, full_s = [], []
    finite = s_probe[~np.isnan(s_probe)]
    floor = float(finite.min()) - 1.0 if len(finite) else -1.0
    for i, k in enumerate(keys):
        if not np.isnan(s_probe[i]):
            full_y.append(int(y[i])); full_s.append(float(s_probe[i]))
    # EVERY rollout the probe could not score, not just the ones with no
    # `</think>`. A rollout can also be absent from the cache because its
    # `</think>` fell outside the truncation window; the previous condition
    # (`... or f["has_think_close"]`) skipped those, so "full population" was
    # still missing a slice — a smaller version of the very filter this section
    # exists to undo. Both reasons are counted separately below.
    n_no_close = n_uncacheable = 0
    for k, f in feats.items():
        if k in cached_keys:
            continue
        full_y.append(int(f["label"])); full_s.append(floor)
        if f["has_think_close"]:
            n_uncacheable += 1
        else:
            n_no_close += 1
    n_added = n_no_close + n_uncacheable
    full_y = np.asarray(full_y); full_s = np.asarray(full_s)
    dep_auroc = balanced_auroc(full_y, full_s)
    res["auroc_deployment_full_population"] = dep_auroc
    res["n_deployment_extra_rows"] = int(n_added)
    res["n_deployment_extra_no_think_close"] = int(n_no_close)
    res["n_deployment_extra_uncacheable"] = int(n_uncacheable)
    L.append("  (4) DEPLOYMENT AUROC — full population, rule: unscoreable => predict wrong")
    L.append(f"      cached-subset AUROC (what the paper reports)   {res['auroc_probe']:.3f}")
    L.append(f"      full-population AUROC (+{n_added} unscored rows)  {dep_auroc:.3f}")
    L.append(f"      of those {n_added}: {n_no_close} had no </think>, "
             f"{n_uncacheable} had one but were not cacheable (truncation)")
    L.append(f"      note: the uncached rows are {100*n_added/max(1,len(feats)):.1f}% of rollouts and "
             f"~100% wrong, so the free rule helps the WORSE checkpoint more.")
    L.append("")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--ckpt", default=None, help="restrict to one checkpoint")
    ap.add_argument("--eval_sft", default=DEFAULT_EVALS["C_SFT"])
    ap.add_argument("--eval_outcome", default=DEFAULT_EVALS["C_outcome"])
    ap.add_argument("--n_boot", type=int, default=10000,
                    help="prompt-clustered bootstrap resamples for section (2b)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    evals = {"C_SFT": args.eval_sft, "C_outcome": args.eval_outcome}
    if args.ckpt:
        evals = {args.ckpt: evals[args.ckpt]}

    L = ["Structural controls for the </think> correctness probe",
         f"  cache: {args.cache_dir}   layer: L{args.layer}   K: {args.K}",
         "",
         "  Question: how much of the probe's AUROC and of its best-of-K lift is",
         "  available from WHERE the model closed <think>, with no probe at all?",
         ""]
    results = []
    for ckpt, path in evals.items():
        r = analyse(ckpt, path, args.cache_dir, args.layer, args.K, L,
                    n_boot=args.n_boot)
        if r:
            results.append(r)

    if len(results) == 2:
        a, b = results
        L.append("### Delta across checkpoints (does RL strengthen the probe, "
                 "or just the length signal?)")
        for key, name in [("auroc_probe", "probe AUROC (cached subset)"),
                          ("auroc_deployment_full_population", "probe AUROC (full population)"),
                          ("auroc_thinkclose_position", "position-only AUROC"),
                          ("auroc_probe_stratified_by_position", "probe AUROC | matched position")]:
            L.append(f"  {name:<38} {a['ckpt']} {a[key]:.3f} -> {b['ckpt']} {b[key]:.3f}"
                     f"   ({b[key]-a[key]:+.3f})")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    with open(args.out.replace(".txt", ".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
