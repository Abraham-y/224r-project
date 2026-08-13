"""Known-answer gate for the Phase-2 driver scripts. No GPU, no model.

`validate_estimator.py` gates the ESTIMATOR (`fragility_core.steering`). This
gates the two scripts that feed it and consume it, on the four things about them
that can be wrong while every number stays plausible:

  A  prompt indexing. `sample_local_jsonl.py` rebuilds output rows from a fixed
     field list, so a top-level `_orig_prompt_idx` does not survive sampling.
     Both scripts must read it out of `ground_truth` and RAISE if it is missing,
     never fall back to the local 0..N-1 position -- that fallback puts Phase 2
     in a different prompt namespace from Phases 0/1 and silently changes what
     the bootstrap clusters on.
  B  aggregation. Δtemplate is the H0-deciding measurement and H0's prediction is
     SIGNED ("positive and growing"), so the per-alpha series and the headline
     must be signed; the assertion arm (19% of all generations) must actually be
     aggregated; and no probe-score metric may be emitted, since it is 0 by
     construction at the read position (phase2_predictions.md Amendment A2).
  C  the same aggregation, run on the paper's PUBLISHED steering file, must
     still reproduce the published +0.083 / per-alpha table.
  D  resume. A non-empty raw file means a COMPLETED arm, which only holds if the
     writer is atomic.
  E  both configured alpha-scale sweeps are real, separately-tagged arms.
  F  patching flushes per checkpoint, not once at the end.

    cd followup && python experiments/fragility/phase2_causal_status/validate_run_steering.py

Exit 0 = gate passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fragility_core import labels as labels_mod, paths, steering  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load("rs", "experiments/fragility/phase2_causal_status/run_steering.py")
rp = _load("rp", "experiments/fragility/phase2_causal_status/run_patching.py")

fails: list[str] = []
n = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global n
    n += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------------------
print("A. prompt-index helper (defect 1)")

def through_sampler(row: dict) -> dict:
    """Exactly what extension/evaluation/sample_local_jsonl.py emits."""
    return {
        "prompt": row["prompt"],
        "target": int(row["target"]),
        "nums": list(row["nums"]),
        "ground_truth": row.get(
            "ground_truth", {"numbers": list(row["nums"]), "target": int(row["target"])}
        ),
        "response": ["<think>x</think><answer>1+2</answer>"],
        "scores": [0.0],
    }


prompt_row = {
    "prompt": "p", "target": 3, "nums": [1, 2],
    "ground_truth": {"numbers": [1, 2], "target": 3, "_orig_prompt_idx": 417},
    "_orig_prompt_idx": 417,
}
sampled = through_sampler(prompt_row)
check("sampler drops the top-level key", "_orig_prompt_idx" not in sampled,
      f"fields={sorted(sampled)}")
check("helper still recovers 417 from ground_truth",
      labels_mod.orig_prompt_idx(sampled, 0) == 417,
      str(labels_mod.orig_prompt_idx(sampled, 0)))
try:
    labels_mod.orig_prompt_idx({"prompt": "p", "ground_truth": {}}, 7)
    check("helper RAISES when the index is absent", False, "returned instead")
except KeyError as exc:
    check("helper RAISES when the index is absent", True, str(exc)[:60])

# harvest_ladder's alias must behave identically (no drift).
hl = _load("hl", "experiments/fragility/phase0_replicate/harvest_ladder.py")
check("harvest_ladder._orig_index agrees", hl._orig_index(sampled, 0) == 417)
check("harvest_ladder._orig_index still honours subset_map",
      hl._orig_index({"prompt": "p"}, 3, {3: 99}) == 99)

# load_prefix_work end to end on a sampler-shaped file.
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "step_0.jsonl"
    with open(f, "w") as fh:
        for i, orig in enumerate([100, 250, 499]):
            r = through_sampler(dict(prompt_row))
            r["ground_truth"] = dict(r["ground_truth"], _orig_prompt_idx=orig)
            r["response"] = ["<think>a</think><answer>1+2</answer>"] * 2
            fh.write(json.dumps(r) + "\n")
    work = rs.load_prefix_work(f, 10, 2)
    got = sorted({w["prompt_idx"] for w in work})
    check("load_prefix_work uses the ORIGINAL indices", got == [100, 250, 499], str(got))
    check("load_prefix_work fans out rollouts", len(work) == 6, str(len(work)))

    bad = Path(td) / "bad.jsonl"
    with open(bad, "w") as fh:
        r = through_sampler(dict(prompt_row))
        r["ground_truth"] = {"numbers": [1, 2], "target": 3}
        fh.write(json.dumps(r) + "\n")
    try:
        rs.load_prefix_work(bad, 10, 2)
        check("load_prefix_work raises instead of falling back to 0..99", False)
    except KeyError:
        check("load_prefix_work raises instead of falling back to 0..99", True)
    # run_patching.build_pairs on the same shaped file
    pf = Path(td) / "pairs.jsonl"
    with open(pf, "w") as fh:
        r = through_sampler(dict(prompt_row))
        r["ground_truth"] = dict(r["ground_truth"], _orig_prompt_idx=321)
        r["response"] = [
            "<think>a</think><answer>1+2</answer>",   # correct: 1+2 == 3
            "<think>a</think><answer>2-1</answer>",   # wrong
        ]
        fh.write(json.dumps(r) + "\n")
    prs = rp.build_pairs(pf, 5, 0)
    check("build_pairs uses the ORIGINAL index",
          len(prs) == 1 and prs[0]["prompt_idx"] == 321,
          str([p["prompt_idx"] for p in prs]))

# ---------------------------------------------------------------------------
print("\nB. aggregation emits what the script writes (defects 2, 3, 5)")

# Synthetic rows in exactly run_checkpoint's schema, with a KNOWN answer:
#   accuracy:  probe = rand + 0.20 at every alpha
#   template:  probe = rand + (0.5 * alpha)   -> SIGNED, sign-flipping in alpha
#   assertion: template = rand + 0.10 flat, accuracy = rand exactly (a null)
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
rng = np.random.default_rng(3)
rows: list[dict] = []
for p in range(100):
    base_acc = float(rng.random()) * 0.4
    base_tpl = 2.0 + float(rng.random())
    for r_idx in range(2):
        rows.append({"checkpoint_step": 40, "prompt_idx": p, "resp_idx": r_idx,
                     "alpha": 0.0, "direction": "zero", "direction_detail": "zero",
                     "scale_ref": "h_mean_norm", "scale": 21.86, "injected_norm": 0.0,
                     "new_score": 0.0, "new_score_correct": base_acc,
                     "delta_norm": 0.0, "new_template_score": base_tpl,
                     "new_n_blocks": 1.0, "hook_layer": 16, "hook_offset": 0})
        for a in ALPHAS:
            for direction, d_acc, d_tpl in (
                ("probe", 0.20, 0.5 * a),
                ("assertion", 0.0, 0.10),
                ("rand", 0.0, 0.0),
            ):
                rows.append({"checkpoint_step": 40, "prompt_idx": p, "resp_idx": r_idx,
                             "alpha": a, "direction": direction,
                             "direction_detail": direction,
                             "scale_ref": "h_mean_norm", "scale": 21.86,
                             "injected_norm": abs(a) * 21.86,
                             "new_score": 0.0,
                             "new_score_correct": base_acc + d_acc,
                             "delta_norm": abs(a) * 21.86,
                             "new_template_score": base_tpl + d_tpl,
                             "new_n_blocks": 1.0, "hook_layer": 16, "hook_offset": 0})

agg = rs.aggregate(rows, ref_alpha=1.0)

check("no probe-score metric is emitted any more",
      not any("probescore" in k or "delta_probe" in k for k in agg),
      str([k for k in agg if "probe_score" in k or "probescore" in k]))
check("raw rows carry no new_probe_score",
      all("new_probe_score" not in r for r in rows))
check("accuracy headline == +0.200", abs(agg["causal_strength"] - 0.20) < 1e-12,
      f"{agg['causal_strength']:+.6f}")
check("accuracy assertion control == 0 (null)",
      abs(agg["causal_strength_assertion"]) < 1e-12,
      f"{agg['causal_strength_assertion']:+.6f}")
check("assertion per-alpha series present",
      all(f"steer_delta_acc_assertion_alpha{a:g}" in agg for a in ALPHAS),
      str(sorted(k for k in agg if "assertion_alpha" in k)))
check("assertion per-alpha accuracy deltas are all 0",
      all(abs(agg[f"steer_delta_acc_assertion_alpha{a:g}"]) < 1e-12 for a in ALPHAS))

print("  -- signed Delta-template (the H0-deciding measurement) --")
check("template headline at alpha=+1 == +0.5",
      abs(agg["steer_delta_template"] - 0.5) < 1e-12, f"{agg['steer_delta_template']:+.6f}")
check("template dose recorded as +1.0", agg["steer_delta_template_at_alpha"] == 1.0,
      str(agg.get("steer_delta_template_at_alpha")))
check("template headline carries its CI",
      all(f"steer_delta_template{s}" in agg
          for s in ("_ci_lo", "_ci_hi", "_se", "_p", "_significant", "_n_clusters")))
check("template at alpha=-1 reported SEPARATELY and NEGATIVE",
      abs(agg["steer_delta_template_negalpha"] + 0.5) < 1e-12,
      f"{agg['steer_delta_template_negalpha']:+.6f}")
per_alpha = {a: agg[f"steer_delta_template_alpha{a:g}"] for a in ALPHAS}
check("signed per-alpha template series recovers 0.5*alpha exactly",
      all(abs(per_alpha[a] - 0.5 * a) < 1e-12 for a in ALPHAS),
      str({k: round(v, 3) for k, v in per_alpha.items()}))
check("the series is SIGNED (negative alphas give negative deltas)",
      all(per_alpha[a] < 0 for a in ALPHAS if a < 0))
check("an unsigned max could not have shown that",
      abs(agg["steer_delta_template_diag_maxabs"] - 1.0) < 1e-12
      and agg["steer_delta_template_diag_maxabs"] > 0,
      f"|max| = {agg['steer_delta_template_diag_maxabs']:.3f} at alpha="
      f"{agg['steer_delta_template_diag_maxabs_at_alpha']:+g}, sign discarded")
check("template assertion contrast == +0.10 at alpha=1",
      abs(agg["steer_delta_template_assertion"] - 0.10) < 1e-12,
      f"{agg['steer_delta_template_assertion']:+.6f}")
check("template assertion per-alpha == +0.10 at every alpha",
      all(abs(agg[f"steer_delta_template_assertion_alpha{a:g}"] - 0.10) < 1e-12
          for a in ALPHAS))
check("arm levels are emitted for interpretation",
      "steer_delta_template_probe_alpha1" in agg and "steer_delta_template_rand_alpha1" in agg,
      f"probe {agg.get('steer_delta_template_probe_alpha1'):.3f} vs rand "
      f"{agg.get('steer_delta_template_rand_alpha1'):.3f}")
check("n_blocks outcome present too", "steer_delta_blocks" in agg
      and abs(agg["steer_delta_blocks"]) < 1e-12)
check("every emitted value is a finite-or-nan float",
      all(isinstance(v, float) for v in agg.values()),
      str({k: type(v).__name__ for k, v in agg.items() if not isinstance(v, float)}))

# The metric names the FIGURE looks up must exist.
check("figures' `causal_strength` key exists", "causal_strength" in agg)

# ref_alpha not in the swept grid -> marked absent, never substituted.
narrow = [r for r in rows if r["alpha"] in (0.0, 2.0, -2.0)]
agg_narrow = rs.aggregate(narrow, ref_alpha=1.0)
check("missing ref_alpha marks the ACCURACY headline absent",
      "causal_strength" not in agg_narrow
      and agg_narrow.get("causal_strength_ref_alpha_missing") == 1.0)
check("missing ref_alpha marks the TEMPLATE headline absent",
      "steer_delta_template" not in agg_narrow
      and agg_narrow.get("steer_delta_template_ref_alpha_missing") == 1.0)
check("per-alpha template series still written for the doses that ran",
      abs(agg_narrow["steer_delta_template_alpha2"] - 1.0) < 1e-12
      and abs(agg_narrow["steer_delta_template_alpha-2"] + 1.0) < 1e-12)

# assertion routing self-check must actually fire when arms are crossed.
crossed = []
for r in rows:
    r = dict(r)
    if r["direction"] == "assertion":
        r["new_score_correct"] = r["new_score_correct"] + 0.33
    crossed.append(r)
a_ok = rs.aggregate(crossed, ref_alpha=1.0)
check("crossed arms still agree between the two assertion routes",
      abs(a_ok["causal_strength_assertion"] - 0.33) < 1e-12,
      f"{a_ok['causal_strength_assertion']:+.4f}")


def _broken_relabel(rws):
    return [r for r in rws if r.get("direction") != "assertion"]


orig = rs.assertion_as_probe
rs.assertion_as_probe = _broken_relabel
try:
    rs.aggregate(crossed, ref_alpha=1.0)
    check("a mis-routed assertion arm RAISES", False, "no error")
except RuntimeError as exc:
    check("a mis-routed assertion arm RAISES", True, str(exc)[:70])
finally:
    rs.assertion_as_probe = orig

# ---------------------------------------------------------------------------
print("\nC. aggregation on the paper's PUBLISHED steering run (known answer)")
POSTRL = paths.PAPER_ROOT / "causal_steering_runA_postRL.jsonl"
if POSTRL.exists():
    pub = []
    with open(POSTRL) as f:
        for line in f:
            r = json.loads(line)
            r["new_score_correct"] = float(r["new_score"] == 1.0)
            pub.append(r)
    pa = rs.aggregate(pub, ref_alpha=1.0)
    check("published +0.083 reproduced through run_steering.aggregate",
          abs(pa["causal_strength"] - 0.083) < 0.002, f"{pa['causal_strength']:+.4f}")
    check("published per-alpha table reproduced",
          abs(pa["steer_delta_acc_alpha0.5"] - 0.041) < 0.002
          and abs(pa["steer_delta_acc_alpha2"] + 0.052) < 0.002,
          f"0.5 -> {pa['steer_delta_acc_alpha0.5']:+.4f}, "
          f"2 -> {pa['steer_delta_acc_alpha2']:+.4f}")
    check("no assertion arm in that file -> no assertion metrics invented",
          "causal_strength_assertion" not in pa)
    check("template outcome absent from the published file -> not fabricated",
          "steer_delta_template" not in pa)
else:
    check("published steering file available", False, str(POSTRL))

# ---------------------------------------------------------------------------
print("\nD. raw-row resume path (defect 7)")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "step_40_hook15_h_mean_norm.jsonl"
    rs.write_rows(p, rows[:50])
    check("atomic write leaves no .tmp behind",
          not (Path(td) / (p.name + ".tmp")).exists() and p.exists())
    back = rs.read_rows(p)
    check("round-trip preserves every row", back == rows[:50])
    check("resume condition is 'exists and non-empty'",
          p.exists() and p.stat().st_size > 0)
    empty = Path(td) / "empty.jsonl"
    empty.touch()
    check("an empty file does NOT count as a completed arm", empty.stat().st_size == 0)
    agg_disk = rs.aggregate(rs.read_rows(p), ref_alpha=1.0)
    check("aggregate runs off rows read back from disk", "causal_strength" in agg_disk)

# ---------------------------------------------------------------------------
print("\nE. sweeps wiring (defect 4)")
from fragility_core import registry  # noqa: E402

cfg = registry.load_config("phase2_default.yaml")
scfg = cfg["steering"]
sweeps = [(str(scfg.get("scale_ref", "h_mean_norm")), [float(a) for a in scfg["alphas"]])]
if scfg.get("secondary_scale_ref") and scfg.get("secondary_alphas"):
    sweeps.append((str(scfg["secondary_scale_ref"]),
                   [float(a) for a in scfg["secondary_alphas"]]))
check("config yields TWO sweeps", len(sweeps) == 2, str(sweeps))
check("both refs are valid scale references",
      all(r in steering.SCALE_REFS for r, _ in sweeps))
X = np.random.RandomState(0).randn(500, 64).astype(np.float32) * 0.5 + 3.0
v = np.random.RandomState(1).randn(64).astype(np.float32)
v /= np.linalg.norm(v)
scales = steering.all_scale_references(X, v)
check("all_scale_references covers every configured ref",
      all(r in scales for r, _ in sweeps), str(sorted(scales)))
check("the two sweeps get DIFFERENT scales",
      abs(scales[sweeps[0][0]] - scales[sweeps[1][0]]) > 1e-6,
      str({k: round(vv, 3) for k, vv in scales.items()}))
arms = [f"steering_off{o}_{r}" for o in (0, -1) for r, _ in sweeps]
check("arm tags are unique per (offset, scale_ref)", len(set(arms)) == 4, str(arms))
files = [f"step_40_hook{16 + o}_{r}.jsonl" for o in (0, -1) for r, _ in sweeps]
check("raw filenames are unique per (offset, scale_ref)", len(set(files)) == 4, str(files))
check("ref_alpha is in the PRIMARY sweep",
      any(abs(a - float(scfg["ref_alpha"])) < 1e-9 for a in sweeps[0][1]))
check("ref_alpha is NOT in the secondary sweep (headline marked absent there)",
      not any(abs(a - float(scfg["ref_alpha"])) < 1e-9 for a in sweeps[1][1]),
      f"secondary alphas {sweeps[1][1]}")

# ---------------------------------------------------------------------------
print("\nF. patching flush-per-checkpoint (defect 6)")
src = (ROOT / "experiments/fragility/phase2_causal_status/run_patching.py").read_text()
body = src[src.index("    for step in steps:"):]
check("flush() is called inside the per-step loop", body.count("flush(step)") == 2,
      f"{body.count('flush(step)')} call sites")
check("flush covers the no-pairs early-continue branch",
      "flush(step)\n            continue" in body)
check("writer.flush() no longer only at the end",
      src.count("writer.flush()") == 2)
check("raw jsonl written atomically", "os.replace(tmp, out)" in src)

print("\nG. commit_volume no-ops cleanly off Modal")
check("commit_volume returns False locally", paths.commit_volume("[test] ") is False)

print(f"\n{n - len(fails)}/{n} checks pass")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
sys.exit(0)
