"""The missing cell of the weights x text design: OLD weights on NEW text.

The project has measured three cells:

    (w0, t0)   step-0 weights, step-0 text     baseline
    (wt, t0)   step-t weights, step-0 text     "fixed_text" arm
    (wt, tt)   step-t weights, step-t text     "on_policy" arm

and attributed the monitor's failure between weights and text by subtracting the
second from the third. That subtraction is not identified: the two are measured
on different row sets, and the fourth cell is free to take any value in
[0.5, 0.796], which lets the behavioural share span [0%, 136%].

This script measures the fourth cell:

    (w0, tt)   step-0 weights, step-t text     THIS SCRIPT

It is the direct test. Holding weights at the pre-attack checkpoint and feeding
the attacked policy's own text asks: does that text defeat the monitor on its
own, or only once run through the weights it was co-optimised with?

Sharp predictions, fixed before running:
  - additive / pure-distribution-shift: AUROC(w0,t99) ~ 0.615, and the
    UNOPTIMISED step-0 model already scores step-99 text at mean probe ~ 0.999.
  - interaction / co-adaptation:        AUROC(w0,t99) ~ 0.75-0.80, mean probe ~ 0.5.

Snapshots are written under run_id `{base}__frozen_w0`, with `checkpoint_step`
naming the step whose TEXT was used (not the weights, which are always step 0).
The manifest records both explicitly so the convention cannot be misread.

Three things this script must not do quietly, because it produced a headline
number:

  - Fall back to the unfiltered prompt file. `harvest_ladder.build_prompt_subset`
    only WARNS when the contamination manifest is missing. With this config
    (clean_only, max_prompts 406) the fallback then yields the FIRST 406 prompts
    of the unfiltered eval file: 406 rows numbered 0..405, of which 77 are
    contaminated prompts the clean subset excludes (measured, not inferred). It
    looks exactly like the clean-406 subset and is not it, and nothing
    downstream would notice. The manifest is required here, and the map is
    checked against the clean-406 index set before any GPU work starts.
  - Write snapshots with no provenance. The probe pickle's identity and the run
    config now ride in the per-snapshot manifests and in a run-level manifest,
    as `harvest_ladder` already does.
  - Lose the whole run to one bad text step. Each step is wrapped, its outcome
    recorded in the report, and the registry row is written whatever happens.

    modal run --detach modal_fragility.py cross_cell -- \
        --config phase0_harvest_runA.yaml --text_steps 50,70,99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import activations, checkpoint_logging, labels as labels_mod, paths, probes, registry  # noqa: E402

from harvest_ladder import (  # noqa: E402
    _commit_volume,
    _resolve,
    build_labels,
    build_prompt_subset,
    extract_activations,
    sanity_check_probe,
)
# One definition of the namespace string, shared with the repair tool that gates
# on it.
from relabel_prompt_index import NAMESPACE_ORIG  # noqa: E402

# The contamination filter keeps 406 of the 500 eval prompts, and every cell of
# the weights x text design was measured on exactly that subset. A different
# count means this cell is not comparable to the other three.
EXPECTED_N_PROMPTS = 406


def build_checked_prompt_subset(cfg: dict, prompts_jsonl: Path) -> tuple[int, dict[int, int]]:
    """Write the prompt subset and return (n_prompts, local -> original index).

    Raises rather than falling back. `build_prompt_subset` degrades to the
    unfiltered prompt file with a warning when the contamination manifest is
    missing; here that would silently change the prompt set of a headline
    measurement while leaving it the same size.
    """
    ev = cfg["eval_set"]
    clean = labels_mod.clean_prompt_idx()
    if ev.get("clean_only", True) and clean is None:
        raise SystemExit(
            "contamination manifest missing (extension/data/contaminated_prompt_idx.json) "
            "but eval_set.clean_only is true. Refusing to harvest: the fallback would "
            "silently take the first max_prompts rows of the UNFILTERED eval file, "
            "which is not the subset the other three cells of this design were "
            "measured on."
        )

    n_prompts = build_prompt_subset(cfg, prompts_jsonl)
    subset_map = {
        i: int((r.get("ground_truth") or {}).get("_orig_prompt_idx", i))
        for i, r in enumerate(labels_mod.load_eval_rows(prompts_jsonl))
    }
    if len(subset_map) != EXPECTED_N_PROMPTS:
        raise SystemExit(
            f"prompt subset has {len(subset_map)} prompts, expected {EXPECTED_N_PROMPTS}. "
            "The other cells of the weights x text design were measured on the "
            "clean-406 subset; a different subset is not comparable."
        )
    if clean is not None and not set(subset_map.values()) <= clean:
        stray = sorted(set(subset_map.values()) - clean)[:5]
        raise SystemExit(
            f"prompt subset carries indices outside the clean set (first: {stray}); "
            "prompt_idx would not be in the original eval file's namespace."
        )
    return n_prompts, subset_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text_steps", default="99",
                    help="comma-separated steps whose ROLLOUT TEXT to use")
    ap.add_argument("--weights_step", type=int, default=0,
                    help="the checkpoint whose WEIGHTS are held fixed")
    ap.add_argument("--work_dir", default="/vol/fragility")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    registry.set_all_seeds(int(cfg.get("seed", 0)))

    import os

    work = Path(args.work_dir)
    os.environ.setdefault("FRAGILITY_ACTS_DIR", str(work / "acts"))
    paths.mkdirs()

    base = cfg["name"]
    out_run = f"{base}__frozen_w{args.weights_step}"
    rollout_dir = work / "rollouts" / base

    # The subset map lets rollouts sampled before the prompt-index fix resolve to
    # original indices (see harvest_ladder._orig_index).
    prompts_jsonl = work / "prompts" / f"{base}.jsonl"
    n_prompts, subset_map = build_checked_prompt_subset(cfg, prompts_jsonl)

    ladder = {c.step: c for c in checkpoint_logging.resolve_ladder(cfg["ladder"])}
    if args.weights_step not in ladder:
        raise SystemExit(f"weights step {args.weights_step} not in the ladder")
    model_path = ladder[args.weights_step].model_path

    frozen, prov = probes.load_frozen_probe(_resolve(cfg["probe"]["pipeline"]))
    probe_layer = int(cfg["probe"]["layer"])
    layers = [int(L) for L in cfg["layers"]]
    label_rule = cfg.get("label_rule", "last_block")
    max_seq_len = int(cfg.get("max_seq_len", 2048))

    print(f"[cross] weights held at step {args.weights_step}: {model_path}")

    report: list[dict] = []

    for step_s in args.text_steps.split(","):
        text_step = int(step_s)
        rec: dict = {"text_step": text_step, "weights_step": args.weights_step}
        rj = rollout_dir / f"step_{text_step}.jsonl"
        if not rj.exists():
            print(f"[cross] text step {text_step}: no rollouts at {rj}; skipped")
            rec["status"] = f"skipped: no rollouts at {rj}"
            report.append(rec)
            continue
        print(f"\n[cross] === w{args.weights_step} x t{text_step} ===")

        # One failing text step must not cost the steps after it, nor the
        # registry row. Same handling as harvest_ladder's per-checkpoint arms.
        try:
            X_by_layer, meta = extract_activations(
                model_path, rj, layers, max_seq_len=max_seq_len, subset_map=subset_map
            )
            lab = build_labels(rj, meta, subset_map)
            rep = sanity_check_probe(frozen, X_by_layer[probe_layer], lab, label_rule)

            activations.save_snapshot(
                out_run, text_step, X_by_layer, lab,
                manifest={
                    "arm": "frozen_weights",
                    "weights_step": args.weights_step,
                    "weights_model_path": model_path,
                    "text_step": text_step,
                    "rollouts": str(rj),
                    "n_prompts": n_prompts,
                    "probe_provenance": prov,
                    "sampling": cfg["eval_set"],
                    # prompt_idx comes from labels.orig_prompt_idx via subset_map,
                    # whose values were checked against the clean-406 index set
                    # above, so it indexes the original 500-prompt eval file.
                    "prompt_idx_namespace": NAMESPACE_ORIG,
                    "note": (
                        "checkpoint_step names the TEXT's step; the weights are held "
                        f"at step {args.weights_step}"
                    ),
                },
                overwrite=args.overwrite,
            )
            print(f"[cross]   AUROC(w{args.weights_step}, t{text_step}) = {rep['auroc']:.4f}  "
                  f"mean_probe = {rep['probe_mean']:.4f}")
            rec["status"] = "ok"
            rec["probe"] = rep
        except Exception as exc:
            print(f"[cross]   TEXT STEP {text_step} FAILED: {exc}")
            rec["status"] = f"failed: {exc}"
        report.append(rec)
        _commit_volume()

    if activations.list_steps(out_run):
        activations.save_run_manifest(
            out_run,
            {
                "run_id": out_run,
                "base_run_id": base,
                "arm": "frozen_weights",
                "weights_step": args.weights_step,
                "weights_model_path": model_path,
                "text_steps": args.text_steps,
                "n_prompts": n_prompts,
                "prompt_idx_namespace": NAMESPACE_ORIG,
                "probe_provenance": prov,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "report": report,
                "note": (
                    "checkpoint_step names the TEXT's step; the weights are held "
                    f"at step {args.weights_step}"
                ),
            },
        )

    registry.register(cfg, phase="phase0", kind="cross_cell",
                      weights_step=args.weights_step, text_steps=args.text_steps,
                      report=report)

    ok = sum(1 for r in report if r.get("status") == "ok")
    print(f"\n[cross] snapshots written: {ok}/{len(report)} -> {activations.run_dir(out_run)}")
    if ok < len(report):
        print("[cross] failed/skipped text steps are ABSENT from the cache, not filled in.")


if __name__ == "__main__":
    main()
