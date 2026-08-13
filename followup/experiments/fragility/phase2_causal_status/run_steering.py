"""Phase 2 — activation-addition steering along the t=0 probe direction.

READ phase2_predictions.md FIRST. The predictions are pre-registered; this script
must not be edited to change what is measured after a run has started.

Protocol, per (checkpoint, scale reference, alpha, direction):
  1. prefix = prompt + response up to and including the first `</think>`
  2. hook the decoder layer, add `alpha * scale * v` at the `</think>` position
  3. regenerate the completion
  4. score it: verifier correctness, template score, n_answer_blocks

`direction` is `probe`, `assertion` (correctness-correlated but never optimised)
or `rand`; every reported effect is a contrast against the MATCHED RANDOM arm at
the same alpha and the same injected norm, and both `probe - rand` and
`assertion - rand` are reported through the identical estimator. A bare probe
delta conflates the intervention with the generic damage of perturbing the
residual stream, and the assertion arm is the control that separates "this
direction is causal" from "any correctness-correlated direction is causal".

Two layer conventions are run (see phase2_predictions.md §1a):
  hook_offset=0   hooks layers[L]   -> reproduces the published protocol
  hook_offset=-1  hooks layers[L-1] -> output IS hidden_states[L], the probe's input

Two alpha-scale conventions are run (see phase2_predictions.md §1b): the primary
`steering.scale_ref` sweep (paper-comparable) and, when the config declares
`secondary_scale_ref` / `secondary_alphas`, the secondary sweep. Each is a
separate arm with its own raw-rollout file, tagged `steering_off{offset}_{ref}`.
The pre-registered headline dose `ref_alpha` belongs to the primary sweep only,
so the secondary arm records `causal_strength_ref_alpha_missing` instead of a
headline — never a headline read off a different dose.

Resumable: an arm whose raw rollout file already exists is re-aggregated from
disk instead of regenerated (`--overwrite` forces regeneration). Raw files are
written through a temp file and renamed, so a non-empty file is always complete.

    modal run --detach modal_fragility.py steering -- \
        --config phase2_default.yaml --ladder runA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import (  # noqa: E402
    checkpoint_logging,
    labels as labels_mod,
    metrics_io,
    paths,
    registry,
    steering,
)
from extension.probe.cache_hidden_states import THINK_CLOSE, char_to_token_index  # noqa: E402

# Outcome columns aggregated in addition to accuracy. Δtemplate is the
# H0-discriminating measurement (phase2_predictions.md §2/§3): H0 predicts it is
# POSITIVE AND GROWING, which is a claim about a SIGN, so every one of these is
# reported signed, per alpha and at the pre-registered dose, never as |max|.
EXTRA_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("new_template_score", "steer_delta_template"),
    ("new_n_blocks", "steer_delta_blocks"),
)

# The keys `cluster_bootstrap_*` attaches to a point estimate. Copied as a block
# so a renamed outcome keeps its CI instead of an orphaned number.
_CI_SUFFIXES = ("", "_ci_lo", "_ci_hi", "_se", "_p", "_significant",
                "_n_clusters", "_n_a", "_n_b", "_n_boot_valid")


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.PAPER_ROOT / path


def load_prefix_work(rollouts_json: Path, n_prompts: int, n_per_prompt: int):
    """(prompt, response, target, nums, prompt_idx, resp_idx) for each steered prefix."""
    rows = labels_mod.load_eval_rows(rollouts_json)
    work = []
    for local_idx, row in enumerate(rows[:n_prompts]):
        # ORIGINAL 0..499 index, read out of `ground_truth` (the sampler drops
        # top-level keys), raising if it is absent. It is the bootstrap's cluster
        # key and the join key against every other cache, so a silent fallback to
        # the local 0..99 position would put Phase 2 in a different prompt
        # namespace from Phase 0/1 without changing a single number's plausibility.
        p_idx = labels_mod.orig_prompt_idx(row, local_idx)
        for r_idx, resp in enumerate(row["response"][:n_per_prompt]):
            work.append(
                {
                    "prompt": row["prompt"], "response": resp,
                    "target": int(row["target"]), "nums": list(row["nums"]),
                    "prompt_idx": p_idx, "resp_idx": r_idx,
                }
            )
    return work


# ---------------------------------------------------------------------------
# aggregation (pure; no GPU, no model — runs off the raw rows alone)
# ---------------------------------------------------------------------------


def _copy_ci(src: dict, base: str, dst: dict, name: str) -> bool:
    """Copy `base` and its CI suffixes into `dst` under `name`. False if absent."""
    if base not in src:
        return False
    for suf in _CI_SUFFIXES:
        if base + suf in src:
            dst[name + suf] = src[base + suf]
    if base + "_alpha" in src:
        # The dose, renamed so it cannot be misread as a member of the per-alpha
        # series `{name}_alpha0.5` / `{name}_alpha1` / `{name}_alpha-1` / ...
        dst[name + "_at_alpha"] = src[base + "_alpha"]
    return True


def _copy_per_alpha(src: dict, src_prefix: str, dst: dict, dst_prefix: str) -> None:
    """Copy every `{src_prefix}{alpha}` key across, re-prefixed. Signs preserved."""
    for k, v in src.items():
        if k.startswith(src_prefix):
            dst[dst_prefix + k[len(src_prefix):]] = v


def assertion_as_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Relabel the assertion arm as `probe` and drop the real probe arm.

    `steering.causal_strength` computes its per-alpha contrasts for the `probe`
    arm only, so this feeds the assertion arm through the IDENTICAL code path
    rather than adding a second implementation of the same contrast. The
    `causal_strength` it returns must equal the `causal_strength_assertion` the
    unrelabelled call returns; `aggregate` asserts exactly that.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        d = r.get("direction")
        if d == "probe":
            continue
        if d == "assertion":
            r = dict(r)
            r["direction"] = "probe"
        out.append(r)
    return out


def outcome_metrics(
    rows: list[dict[str, Any]], *, accuracy_key: str, name: str, ref_alpha: float
) -> dict[str, float]:
    """Every metric for one non-accuracy outcome column, namespaced under `name`.

    Every contrast below is SIGNED, because the prediction it tests is signed
    (phase2_predictions.md §3 H0.3: Δtemplate "positive, growing"):

      {name}[_ci_lo/_ci_hi/_se/_p/_significant/...]   probe - rand at ref_alpha
      {name}_at_alpha                                  the dose that was
      {name}_negalpha*                                 same contrast at -ref_alpha
      {name}_assertion*                                assertion - rand at ref_alpha
      {name}_alpha{a}                                  probe - rand at each alpha
      {name}_assertion_alpha{a}                        assertion - rand at each alpha
      {name}_probe_alpha{a} / {name}_rand_alpha{a}     the arm levels behind them
      {name}_ref_alpha_missing                         instead of {name}, if the
                                                       sweep never ran ref_alpha
      {name}_diag_*                                    the demoted diagnostics of
                                                       Amendment A1; `_diag_maxabs`
                                                       is unsigned, which is why it
                                                       is not the headline
    """
    src = steering.causal_strength(
        rows, accuracy_key=accuracy_key, probe_key=None, ref_alpha=ref_alpha
    )
    out: dict[str, float] = {}
    if not src:
        return out
    if not _copy_ci(src, "causal_strength", out, name):
        # ref_alpha absent from this sweep: mark it, never substitute another dose.
        out[f"{name}_ref_alpha_missing"] = float(ref_alpha)
    _copy_ci(src, "causal_strength_negalpha", out, f"{name}_negalpha")
    _copy_ci(src, "causal_strength_assertion", out, f"{name}_assertion")
    _copy_per_alpha(src, "steer_delta_acc_alpha", out, f"{name}_alpha")
    _copy_per_alpha(src, "steer_acc_probe_alpha", out, f"{name}_probe_alpha")
    _copy_per_alpha(src, "steer_acc_rand_alpha", out, f"{name}_rand_alpha")
    for k, tag in (
        ("diag_maxabs_delta", "_diag_maxabs"),
        ("diag_maxabs_delta_signed", "_diag_maxabs_signed"),
        ("diag_maxabs_at_alpha", "_diag_maxabs_at_alpha"),
        ("diag_ols_slope", "_diag_ols_slope"),
        ("diag_ols_slope_dof", "_diag_ols_slope_dof"),
    ):
        if k in src:
            out[name + tag] = src[k]

    a_src = steering.causal_strength(
        assertion_as_probe(rows), accuracy_key=accuracy_key, probe_key=None,
        ref_alpha=ref_alpha,
    )
    _copy_per_alpha(a_src, "steer_delta_acc_alpha", out, f"{name}_assertion_alpha")
    _check_assertion_routing(a_src, out, f"{name}_assertion")
    return out


def _check_assertion_routing(a_src: dict, out: dict, key: str) -> None:
    """The same contrast computed two ways must agree exactly.

    `causal_strength_assertion` (arm selected inside the estimator) and the
    relabelled `causal_strength` (arm selected before the estimator) run over the
    same rows with the same seed, so they are bit-identical unless one of them is
    reading the wrong arm.
    """
    if key in out and "causal_strength" in a_src:
        if abs(float(a_src["causal_strength"]) - float(out[key])) > 1e-12:
            raise RuntimeError(
                f"assertion contrast disagrees with itself: relabelled "
                f"{a_src['causal_strength']:+.6g} vs in-estimator {out[key]:+.6g}; "
                "one of the two is reading the wrong arm"
            )


def aggregate(rows: list[dict[str, Any]], *, ref_alpha: float) -> dict[str, float]:
    """Every metric one steering arm writes, computed from its raw rows.

    Pure function of `rows` — no model, no GPU — so it can be re-run on a raw
    rollout file and unit-tested on synthetic rows.

    `probe_key=None` throughout: see `run_checkpoint` for why no probe score is
    recorded per generation.
    """
    agg = dict(steering.causal_strength(rows, probe_key=None, ref_alpha=ref_alpha))
    a_src = steering.causal_strength(
        assertion_as_probe(rows), probe_key=None, ref_alpha=ref_alpha
    )
    _copy_per_alpha(a_src, "steer_delta_acc_alpha", agg, "steer_delta_acc_assertion_alpha")
    _copy_per_alpha(a_src, "steer_acc_probe_alpha", agg, "steer_acc_assertion_alpha")
    _check_assertion_routing(a_src, agg, "causal_strength_assertion")
    for key, name in EXTRA_OUTCOMES:
        if rows and key not in rows[0]:
            continue
        agg.update(outcome_metrics(rows, accuracy_key=key, name=name, ref_alpha=ref_alpha))
    return agg


# ---------------------------------------------------------------------------
# raw-row I/O (resume support)
# ---------------------------------------------------------------------------


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write via a temp file + rename, so a non-empty `path` is always complete."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def run_checkpoint(
    *,
    model_path: str,
    step: int,
    work: list[dict],
    v_unit: np.ndarray,
    v_assert: np.ndarray | None,
    alphas: list[float],
    scale: float,
    scale_ref: str,
    hook_layer: int,
    hook_offset: int,
    cfg_steer: dict,
    max_seq_len: int,
) -> list[dict]:
    """Every steered generation for one (checkpoint, hook offset, scale ref)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluation.countdown import compute_score

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    alphas = [float(a) for a in alphas]
    if cfg_steer.get("both_signs", True):
        alphas = sorted({a for x in alphas for a in (x, -x)})
    n_rand = int(cfg_steer.get("n_random_seeds", 3))
    d = v_unit.size

    v_probe_t = steering.to_torch(v_unit)
    v_rand_t = [
        steering.to_torch(steering.random_matched_direction(d, seed=1000 + i))
        for i in range(n_rand)
    ]
    v_assert_t = steering.to_torch(v_assert) if v_assert is not None else None

    rows: list[dict] = []
    n_skipped = 0
    checked_position = False
    with steering.intervention(model, hook_layer) as state:
        for w_i, w in enumerate(work):
            full = w["prompt"] + w["response"]
            close_char = full.find(THINK_CLOSE, len(w["prompt"]))
            if close_char < 0:
                continue
            prefix = full[: close_char + len(THINK_CLOSE)]
            enc = tok(prefix, return_offsets_mapping=True, truncation=True,
                      max_length=max_seq_len, return_tensors="pt")
            input_ids = enc["input_ids"].to("cuda")
            offs = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            # Inject where the PROBE READS, not at the end of the prefix.
            # `</think>` is three tokens ('</', 'think', '>'), so the prefix's
            # last token is two positions downstream of the probe's read site,
            # and its identity differs from the calibration sequence ('>' vs
            # '>\n\n'). Steering there tests a direction the probe does not read
            # and returns a null by construction — this is findings.md EXP-21
            # bug #7, which cost the original project a round of invalid runs.
            pos = char_to_token_index(offs, close_char)
            if pos is None:
                # Over-long prefixes get right-truncated, dropping `</think>`
                # entirely. Skip rather than fall back to an arbitrary token.
                n_skipped += 1
                continue
            gt = {"target": w["target"], "numbers": w["nums"]}

            conditions = [("zero", 0.0, None)]
            for a in alphas:
                conditions.append(("probe", a, v_probe_t))
                if v_assert_t is not None:
                    conditions.append(("assertion", a, v_assert_t))
                for i, vr in enumerate(v_rand_t):
                    conditions.append((f"rand{i}", a, vr))

            for direction, alpha, vec in conditions:
                if direction == "zero":
                    state.mode, state.vector, state.position = "off", None, None
                else:
                    state.mode = "add"
                    state.vector = (alpha * scale) * vec
                    state.position = pos
                state.reset()

                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=int(cfg_steer.get("max_new_tokens", 400)),
                        do_sample=True,
                        temperature=float(cfg_steer.get("gen_temperature", 0.6)),
                        top_p=float(cfg_steer.get("gen_top_p", 0.95)),
                        top_k=int(cfg_steer.get("gen_top_k", 20)),
                        pad_token_id=tok.eos_token_id,
                    )
                if direction != "zero":
                    if not state.applied:
                        raise RuntimeError(
                            f"steering hook never fired at step {step} pos {pos}; "
                            "a silent no-op would look exactly like a null result"
                        )
                    intended = abs(alpha) * scale
                    if not (0.9 * intended <= state.delta_norm <= 1.1 * intended):
                        raise RuntimeError(
                            f"injected norm {state.delta_norm:.3f} != intended "
                            f"{intended:.3f} at step {step}; the intervention did not "
                            "land as specified"
                        )
                    if not checked_position:
                        # Once per checkpoint, prove the injection site really is
                        # the probe's read site.
                        decoded = tok.decode(input_ids[0, pos:])
                        if not decoded.startswith(THINK_CLOSE):
                            raise RuntimeError(
                                f"injection position {pos} decodes to {decoded[:20]!r}, "
                                f"expected a prefix starting with {THINK_CLOSE!r}"
                            )
                        print(f"[steer]   position check OK: pos={pos} -> {decoded[:12]!r}")
                        checked_position = True
                new_text = tok.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=False)
                score = float(compute_score(new_text, gt))

                # NO frozen-probe score is recorded per generation, deliberately.
                #
                # The probe reads the FIRST `</think>`, which is the last thing in
                # the prefix — it PRECEDES every token this intervention produced.
                # Scoring it with the hook off (the only non-circular way; with the
                # hook armed the answer is the mechanical identity h + alpha*s*v)
                # reads a hidden state that, under causal attention, is a function
                # of tokens <= that position only, i.e. of the prefix alone. The
                # prefix is byte-identical across `zero`, `probe`, `assertion` and
                # every `rand` condition, and BPE is left-to-right so appending
                # `new_text` cannot change the tokens at or before that position
                # (a prior review reports checking this on 160 real rollouts,
                # 160/160 identical). The "Δ probe score" was therefore 0 by
                # construction rather than by measurement, and the earlier version
                # wrote those zeros into the metrics as though a null had been
                # observed — at the cost of one extra full forward pass per
                # generation.
                #
                # phase2_predictions.md §2 lists Δ(probe score) as a planned
                # outcome; it cannot be taken at this position, because the
                # position is upstream of everything the intervention changed.
                # Recorded as a deviation in phase2_predictions.md Amendment A2.
                # Measuring it would need a probe read AFTER the steered tokens,
                # i.e. a different position rule than the probe was calibrated on,
                # which is a different experiment and not a column here.

                lab = labels_mod.block_labels(new_text, w["target"], w["nums"])
                rows.append(
                    {
                        "checkpoint_step": step,
                        "prompt_idx": w["prompt_idx"], "resp_idx": w["resp_idx"],
                        "alpha": alpha,
                        # rand0/rand1/rand2 collapse to "rand" for aggregation
                        "direction": "rand" if direction.startswith("rand") else direction,
                        "direction_detail": direction,
                        "scale_ref": scale_ref,
                        "scale": float(scale),
                        "injected_norm": abs(alpha) * scale,
                        "new_score": score,
                        "new_score_correct": float(score == 1.0),
                        "delta_norm": float(state.delta_norm),
                        "new_template_score": float(lab["template_score"]),
                        "new_n_blocks": float(lab["n_blocks"]),
                        "hook_layer": hook_layer,
                        "hook_offset": hook_offset,
                    }
                )
            if (w_i + 1) % 20 == 0:
                print(f"[steer]   step {step}: {w_i + 1}/{len(work)} prefixes")

    if n_skipped:
        print(f"[steer]   {n_skipped}/{len(work)} prefixes skipped (no locatable </think>)")
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ladder", required=True, help="name from cfg.ladders (e.g. runA)")
    ap.add_argument("--work_dir", default="/vol/fragility")
    ap.add_argument("--hook_offsets", default="0,-1",
                    help="0 reproduces the paper; -1 hooks the probe's true input layer")
    ap.add_argument("--steps", default=None)
    ap.add_argument("--scale_refs", default=None,
                    help="comma-separated subset of the configured scale references "
                         "(default: every sweep the config declares)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate arms whose raw rollout file already exists")
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    registry.set_all_seeds(int(cfg.get("seed", 42)))

    ladder_entry = next((l for l in cfg["ladders"] if l["name"] == args.ladder), None)
    if ladder_entry is None:
        raise SystemExit(f"--ladder {args.ladder!r} not in config ({[l['name'] for l in cfg['ladders']]})")
    lcfg = registry.load_config(ladder_entry["config"])
    run_id = lcfg["name"]

    work_dir = Path(args.work_dir)
    rollouts = work_dir / "rollouts" / run_id
    if not rollouts.exists():
        raise SystemExit(
            f"no harvested rollouts at {rollouts}. Phase 2 steers the prefixes Phase 0 "
            "sampled; run the harvest first."
        )

    probe_cfg = cfg["probe"]
    # The frozen probe pipeline is NOT loaded here: nothing in this script scores
    # it any more (see run_checkpoint). Only the probe's DIRECTION is used.
    with np.load(_resolve(probe_cfg["direction"])) as dz:
        v_unit = np.asarray(dz["v_unit"], dtype=np.float32)
    scfg = cfg["steering"]
    v_assert = None
    if "assertion" in scfg.get("directions", []):
        assert_path = scfg.get("assertion_direction")
        if assert_path and _resolve(assert_path).exists():
            with np.load(_resolve(assert_path)) as dz:
                v_assert = np.asarray(dz["v_unit"], dtype=np.float32)
            from fragility_core import geometry as _geo
            print(f"[steer] assertion control loaded; cos(probe, assertion) = "
                  f"{_geo.cosine(v_unit, v_assert):+.4f}")
        else:
            raise SystemExit(
                f"directions includes 'assertion' but {assert_path!r} is missing; the "
                "pre-registered specificity control cannot run"
            )
    probe_layer = int(probe_cfg["layer"])
    # The dose at which `causal_strength` is reported. Pre-registered in the
    # config, not chosen after seeing the curve; it must be one of the PRIMARY
    # sweep's alphas or the headline is written as ABSENT rather than moved.
    ref_alpha = float(scfg.get("ref_alpha", steering.CAUSAL_REF_ALPHA))

    # phase2_predictions.md §1(b) pre-registers a secondary alpha-scale sweep.
    # It is a real arm: its own generations, its own raw file, its own `arm` tag.
    # (It used to be assembled into a `sweeps` list that nothing iterated, so the
    # config declared an experiment that silently never ran.)
    sweeps: list[tuple[str, list[float]]] = [
        (str(scfg.get("scale_ref", "h_mean_norm")), [float(a) for a in scfg["alphas"]])
    ]
    if scfg.get("secondary_scale_ref") and scfg.get("secondary_alphas"):
        sweeps.append((str(scfg["secondary_scale_ref"]),
                       [float(a) for a in scfg["secondary_alphas"]]))
    for ref, _ in sweeps:
        if ref not in steering.SCALE_REFS:
            raise SystemExit(
                f"scale reference {ref!r} is not one of {steering.SCALE_REFS}"
            )
    # Checked against the CONFIGURED primary sweep, before any --scale_refs
    # filtering, so the config can never declare a headline dose it does not run.
    if not any(abs(a - ref_alpha) < 1e-9 for a in sweeps[0][1]):
        raise SystemExit(
            f"steering.ref_alpha={ref_alpha} is not in the primary sweep's alphas="
            f"{sweeps[0][1]}; the pre-registered headline dose was never run"
        )
    if args.scale_refs:
        want = [s.strip() for s in args.scale_refs.split(",") if s.strip()]
        unknown = [w for w in want if w not in {r for r, _ in sweeps}]
        if unknown:
            raise SystemExit(
                f"--scale_refs {unknown} are not configured sweeps "
                f"({[r for r, _ in sweeps]})"
            )
        sweeps = [s for s in sweeps if s[0] in want]
        if not sweeps:
            raise SystemExit("--scale_refs selected no sweeps")
    for ref, alist in sweeps:
        if not any(abs(a - ref_alpha) < 1e-9 for a in alist):
            print(
                f"[steer] sweep {ref!r} does not include ref_alpha={ref_alpha}; "
                "it will record `*_ref_alpha_missing` instead of a headline "
                "`causal_strength` (by design — the headline dose is not moved)"
            )

    steps = [int(s) for s in (args.steps.split(",") if args.steps else cfg["checkpoint_steps"])]
    ladder = {c.step: c for c in checkpoint_logging.resolve_ladder(lcfg["ladder"])}
    missing = [s for s in steps if s not in ladder]
    if missing:
        # Fail before loading a model: a partially-resolved ladder silently
        # produced an empty metrics file once already.
        raise SystemExit(
            f"steps {missing} not in the ladder (available: {sorted(ladder)}). "
            "Check `ladder.include_final` / `final_step` in the harvest config."
        )

    writer = metrics_io.MetricWriter(run_id, "phase2", int(cfg.get("seed", 42)))
    raw_dir = work_dir / "steering" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    for offset in [int(o) for o in args.hook_offsets.split(",")]:
        hook_layer = probe_layer + offset
        for step in steps:
            if step not in ladder:
                print(f"[steer] step {step} absent from the ladder; no rows written")
                continue
            rj = rollouts / f"step_{step}.jsonl"
            if not rj.exists():
                print(f"[steer] step {step}: no harvested rollouts; no rows written")
                continue

            work = load_prefix_work(
                rj, int(scfg.get("n_prompts", 100)), int(scfg.get("n_rollouts_per_prompt", 2))
            )
            # The alpha scale must be measured on THIS checkpoint's activations,
            # otherwise "alpha=1.0" means a different perturbation at every step.
            from fragility_core import activations as acts_mod

            try:
                X = acts_mod.load_acts(run_id, step, probe_layer)
                scales = steering.all_scale_references(X, v_unit)
            except FileNotFoundError:
                print(f"[steer] step {step}: no cached activations, cannot set the alpha scale")
                continue

            # Injected L2 norm is the only quantity comparable ACROSS scale
            # conventions, so print it for every sweep before spending the GPU
            # time. phase2_predictions.md §1(b) rejected `act_std` precisely
            # because alpha=8 x act_std is ~3% of the paper's alpha=1.0 and
            # "would return a null by construction"; the same arithmetic has to
            # be visible for whatever the config actually asks for. Reported, not
            # rescaled: the alphas are pre-registered and are not tuned here.
            primary_norm = abs(ref_alpha) * scales[sweeps[0][0]]
            for _ref, _alphas in sweeps:
                norms = steering.alpha_norms(_alphas, scales[_ref])
                print(f"[steer]   {_ref}: scale={scales[_ref]:.4f}, injected L2 norms "
                      f"{ {a: round(nm, 3) for a, nm in norms.items()} }")
                biggest = max(norms.values()) if norms else 0.0
                if primary_norm > 0 and biggest < 0.2 * primary_norm:
                    print(
                        f"[steer]   WARNING: {_ref} sweep's LARGEST dose injects "
                        f"{biggest:.3f}, {100 * biggest / primary_norm:.1f}% of the "
                        f"primary's ref_alpha dose ({primary_norm:.3f}). A null in "
                        "that arm is a null at that magnitude, not a null at the "
                        "paper-comparable magnitude — read them separately."
                    )

            for scale_ref, base_alphas in sweeps:
                scale = scales[scale_ref]
                arm = f"steering_off{offset}_{scale_ref}"
                out = raw_dir / f"step_{step}_hook{hook_layer}_{scale_ref}.jsonl"
                # Each sweep is a full extra pass over `work`; print the cost so a
                # two-sweep config is not a surprise halfway through the job.
                n_signed = len({
                    a for x in base_alphas
                    for a in ((x, -x) if scfg.get("both_signs", True) else (x,))
                })
                n_cond = 1 + n_signed * (
                    1 + int(v_assert is not None) + int(scfg.get("n_random_seeds", 3))
                )
                print(
                    f"[steer] step {step} hook L{hook_layer} (probe reads "
                    f"hidden_states[{probe_layer}]) ref={scale_ref} scale={scale:.3f} "
                    f"alphas={base_alphas} -> {len(work) * n_cond} generations "
                    f"({len(work)} prefixes x {n_cond} conditions) "
                    f"refs={ {k: round(v, 3) for k, v in scales.items()} }"
                )

                if out.exists() and out.stat().st_size > 0 and not args.overwrite:
                    # Raw files are written atomically, so a non-empty one is a
                    # completed arm. Re-aggregated rather than skipped, so the
                    # metrics still get (re)written from it.
                    rows = read_rows(out)
                    print(f"[steer]   raw rows exist, reusing {len(rows)} rows <- {out.name} "
                          "(--overwrite to regenerate)")
                else:
                    rows = run_checkpoint(
                        model_path=ladder[step].model_path, step=step, work=work,
                        v_unit=v_unit, v_assert=v_assert, alphas=base_alphas,
                        scale=scale, scale_ref=scale_ref, hook_layer=hook_layer,
                        hook_offset=offset, cfg_steer=scfg,
                        max_seq_len=int(lcfg.get("max_seq_len", 2048)),
                    )
                    write_rows(out, rows)
                    print(f"[steer]   wrote {len(rows)} raw rows -> {out}")
                if not rows:
                    print(f"[steer]   no rows for step {step} {scale_ref}; nothing aggregated")
                    continue

                agg = aggregate(rows, ref_alpha=ref_alpha)
                # `layer` is the layer the DIRECTION belongs to (the probe's), not
                # the hooked module — otherwise offset 0 and offset -1 land under
                # different layer values and a figure joining phase1 and phase2 on
                # `layer` silently mixes conventions. The hook convention and the
                # alpha-scale convention both live in `arm`.
                for k, v in agg.items():
                    writer.add(k, v, checkpoint_step=step, layer=probe_layer,
                               arm=arm, probe_id=probe_cfg.get("id", ""),
                               hook_offset=offset, hook_layer=hook_layer,
                               scale_ref=scale_ref, scale=float(scale))
                for k, v in scales.items():
                    writer.add(f"scale_ref_{k}", v, checkpoint_step=step, layer=probe_layer,
                               arm=arm, scale_source_layer=probe_layer)
                # Flush per (offset, step, scale_ref): a failure later must not
                # discard hours of completed steering. Shards are separate files,
                # so this is free. The Modal volume is only committed in the
                # launcher's `finally`, which a hard timeout skips — so commit too.
                shard = writer.flush()
                print(f"[steer]   metrics -> {shard}")
                paths.commit_volume("[steer]   ")

    # Usually a no-op: every arm already flushed. Kept so any row added outside
    # the loop still lands.
    p = writer.flush()
    registry.register(cfg, phase="phase2", ladder=args.ladder, steps=steps,
                      scale_refs=[r for r, _ in sweeps])
    print(f"[steer] done; final shard: {p}" if p else "[steer] done (all shards flushed per arm)")
    paths.commit_volume("[steer] ")


if __name__ == "__main__":
    main()
