"""Assert the invariants that would catch a silent data error.

Run this before quoting any number. Each check is something that MUST hold if
the pipeline is correct, and whose violation is otherwise invisible — the numbers
stay plausible, the code raises nothing, and the conclusion is wrong.

Every check here exists because it (or its near neighbour) actually failed at
some point:

  1. Cross-arm step-0 identity. At step 0 the fixed-text arm passes that
     checkpoint's own text through its own weights, so its activations must be
     bit-identical to the on-policy arm's and every metric must agree exactly.
     This failed silently (0.787 vs 0.796) when the on-policy metrics predated a
     prompt-index relabel and the arms were split on different val sets. The
     headline decomposition was computed from the mismatched pair.
  2. Prompt-index namespace. Caches must use the ORIGINAL 0..499 indices, not a
     0..405 subset numbering, or the by-prompt split lands the same problem in
     different folds on different ladders.
  3. Fixed-text label constancy. Labels must be identical at every step; that is
     the entire point of the arm.
  4. Row correspondence in the fixed-text arm, without which CKA is undefined.
  5. Metric freshness. Metrics must be newer than the labels they describe, or
     they were computed against a superseded split.

    python experiments/fragility/verify_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fragility_core import activations, labels as labels_mod, metrics_io  # noqa: E402

ON, FT = "phase0_harvest_runA", "phase0_harvest_runA__fixed_text"
LAYER = 16
failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


print("1. cross-arm step-0 identity")
if activations.list_steps(ON) and activations.list_steps(FT):
    Xa = activations.load_acts(ON, 0, LAYER)
    Xf = activations.load_acts(FT, 0, LAYER)
    check("step-0 activations bit-identical", Xa.shape == Xf.shape and np.array_equal(Xa, Xf),
          f"shapes {Xa.shape} vs {Xf.shape}")
    la, lf = activations.load_labels(ON, 0), activations.load_labels(FT, 0)
    check("step-0 prompt_idx identical", la["prompt_idx"].equals(lf["prompt_idx"]))
    check("step-0 labels identical", la["last_block"].equals(lf["last_block"]))

    m = metrics_io.latest(metrics_io.read_metrics(phase="phase1"))
    for metric in ("auroc_frozen", "auroc_retrained_linear", "sep"):
        def get(rid):
            r = m[(m.run_id == rid) & (m.layer == LAYER) & (m.metric == metric)
                  & (m.checkpoint_step == 0) & (m.seed == 0)]["value"]
            return float(r.iloc[0]) if len(r) else float("nan")
        a, f = get(ON), get(FT)
        # Both computed => must agree exactly. A mismatch means the two arms were
        # analysed against different splits.
        agree = (np.isnan(a) and np.isnan(f)) or abs(a - f) < 1e-9
        check(f"step-0 {metric} agrees across arms", agree, f"{a:.4f} vs {f:.4f}")
else:
    check("both arms present", False, "one of the ladders is not cached locally")

print("\n2. prompt-index namespace")
# EVERY step, not just steps[0]. Sampling one step let a mixed-namespace ladder
# (0..499 at the ends, 0..199 in the middle) pass as clean, which silently puts
# the same problem in different val folds at different checkpoints.
for rid in (ON, FT, "phase0_harvest_runB", "vanilla_rloo_ladder",
            "phase0_harvest_runA__frozen_w0"):
    steps = activations.list_steps(rid)
    if not steps:
        continue
    # The invariant is NAMESPACE, not coverage. A cache built from a 200-prompt
    # eval file legitimately maxes at 199 while still using original indices —
    # an earlier version of this check tested `max > 405` and flagged those as
    # violations, which nearly caused correct data to be "repaired".
    # The real test: every prompt_idx must be a member of the clean-406 set,
    # which is defined in the original 0..499 namespace. A cache in a local
    # 0..N-1 namespace fails this as soon as it contains an index that is
    # contaminated-or-absent upstream.
    clean = labels_mod.clean_prompt_idx()
    if clean is None:
        check(f"{rid} namespace checkable", False, "clean-406 manifest missing")
    else:
        bad = []
        for s_ in steps:
            idx = set(int(v) for v in activations.load_labels(rid, s_)["prompt_idx"].unique())
            stray = idx - clean
            if stray:
                bad.append((s_, sorted(stray)[:5], len(stray)))
        check(f"{rid} prompt_idx ⊆ clean-406 at EVERY step", not bad,
              f"steps with out-of-namespace indices: {bad}")

print("\n3. fixed-text label constancy")
steps = activations.list_steps(FT)
if steps:
    ref = activations.load_labels(FT, steps[0])
    for s in steps[1:]:
        lab = activations.load_labels(FT, s)
        same = (
            len(lab) == len(ref)
            and lab["last_block"].equals(ref["last_block"])
            and lab["prompt_idx"].equals(ref["prompt_idx"])
        )
        if not same:
            check(f"step {s} labels match step {steps[0]}", False)
            break
    else:
        check(f"labels identical across all {len(steps)} fixed-text steps",
              len(steps) > 1, f"only {len(steps)} step(s) — vacuous" if len(steps) <= 1 else "")

print("\n4. fixed-text row correspondence (CKA prerequisite)")
if steps:
    # Row COUNTS are not correspondence. The identity of each row must match.
    ref = activations.load_labels(FT, steps[0])[["prompt_idx", "resp_idx", "tok_idx"]]
    ok = all(
        activations.load_labels(FT, s)[["prompt_idx", "resp_idx", "tok_idx"]].equals(ref)
        for s in steps
    )
    check("(prompt_idx, resp_idx, tok_idx) identical across steps", ok)

print("\n4b. the fixed-text arm really varied the weights")
# THE blind spot: if every step had been harvested with one model, checks 1-4 all
# still pass (labels constant, rows constant, namespace fine, step-0 identity
# fine) and frozen AUROC would be flat — which is the observed result. Nothing
# above distinguishes "weights barely matter" from "we never changed the weights".
if steps:
    mps = [activations.load_manifest(FT, s).get("model_path") for s in steps]
    check("a distinct model_path per step", len(set(mps)) == len(steps),
          f"{len(set(mps))} distinct paths for {len(steps)} steps")
    rolls = {activations.load_manifest(FT, s).get("rollouts") for s in steps}
    check("one constant rollouts file across steps", len(rolls) == 1, str(rolls))
    # Data-layer version of the same check: activations must actually differ.
    X0 = activations.load_acts(FT, steps[0], LAYER)
    identical = [s for s in steps[1:] if np.array_equal(activations.load_acts(FT, s, LAYER), X0)]
    check("activations differ from step 0 at every later step", not identical,
          f"bit-identical to step 0 at steps {identical}")

print("\n5. metric freshness vs labels")
m = metrics_io.latest(metrics_io.read_metrics(phase="phase1"))
if "written_utc" in m.columns:
    import datetime as _dt
    for rid in (ON, FT):
        st = activations.list_steps(rid)
        if not st:
            continue
        lab_mtime = max(activations.labels_path(rid, x).stat().st_mtime for x in st)
        rows = m[(m.run_id == rid) & m["written_utc"].notna()]
        if rows.empty:
            continue
        newest = max(
            _dt.datetime.fromisoformat(t).timestamp() for t in rows["written_utc"]
        )
        check(f"{rid} metrics newer than its labels", newest > lab_mtime,
              "" if newest > lab_mtime else "metrics predate the labels: recompute Phase 1")

print(f"\n{checks - len(failures)}/{checks} invariants hold")
if failures:
    print("\nVIOLATIONS — do not quote numbers until these are resolved:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All invariants hold.")
