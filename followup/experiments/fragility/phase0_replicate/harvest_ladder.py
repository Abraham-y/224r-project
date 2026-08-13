"""Phase 0 — harvest activations off an EXISTING probe-RL checkpoint ladder.

Runs inside a Modal container with the training volume mounted at /vol.

Per checkpoint, two arms:

  on_policy   sample fresh rollouts FROM that checkpoint on the fixed prompt set,
              score them with the real verifier, then forward-pass to cache the
              `</think>` activation. This is what the monitor actually sees.
  fixed_text  forward-pass the SAME text at every step (checkpoint-0's rollouts)
              through checkpoint t. Labels are then constant by construction, so
              any movement is representation drift rather than behaviour drift.
              This is also the only arm where CKA is defined.

Both arms are needed, and neither alone answers the question. On-policy alone
cannot separate "the representation moved" from "the policy now writes different
text"; fixed-text alone cannot tell you whether the deployed monitor still works.

Resumable: a checkpoint whose rollouts and snapshot already exist is skipped, so
a timed-out job can be relaunched without repeating GPU work. That only holds if
the volume is committed as we go — Modal's launcher commits in a `finally`, i.e.
once at job end, so a crash on checkpoint 9 of 11 would otherwise discard all
nine. `_commit_volume` is called after every checkpoint.

Activation extraction is IMPORTED from `extension/probe/cache_hidden_states.py`,
never reimplemented. Two historical bugs (findings.md EXP-21 #7 and #8) came from
a second implementation drifting from the cache's conventions, and both produced
a saturated probe rather than an error.

    modal run modal_fragility.py harvest -- --config phase0_harvest_runA.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fragility_core import (  # noqa: E402
    activations,
    checkpoint_logging,
    labels as labels_mod,
    paths,
    probes,
    registry,
)

# The reference implementation. Imported, not copied.
from extension.probe.cache_hidden_states import (  # noqa: E402
    THINK_CLOSE,
    char_to_token_index,
)

_SAMPLER = paths.PAPER_ROOT / "extension" / "evaluation" / "sample_local_jsonl.py"


def _orig_index(row: dict, local_idx: int, subset_map: dict[int, int] | None = None) -> int:
    """The prompt's index in the ORIGINAL eval file, not this subset's ordering.

    Thin alias for `labels.orig_prompt_idx`, which carries the rule and the
    reason. Phase 2 needs the same rule, and a second copy of it there is exactly
    how the harvested caches ended up in a different index namespace from the
    original project's.
    """
    return labels_mod.orig_prompt_idx(row, local_idx, subset_map)


def _commit_volume() -> None:
    """Flush the Modal volume so finished checkpoints survive a later crash.

    No-op outside a Modal container (import fails or the volume is unknown), so
    the harvester still runs locally.
    """
    try:
        import modal

        modal.Volume.from_name(
            os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
        ).commit()
        print("[harvest]   volume committed", flush=True)
    except Exception as exc:
        print(f"[harvest]   volume commit skipped ({exc})", flush=True)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.PAPER_ROOT / path


# ---------------------------------------------------------------------------
# prompt set
# ---------------------------------------------------------------------------


def build_prompt_subset(cfg: dict, out_path: Path) -> int:
    """Write the fixed, contamination-filtered prompt subset used at every step."""
    ev = cfg["eval_set"]
    rows = labels_mod.load_eval_rows(_resolve(ev["prompts"]))
    clean = labels_mod.clean_prompt_idx() if ev.get("clean_only", True) else None
    if ev.get("clean_only", True) and clean is None:
        print("[harvest] WARNING: clean-406 manifest missing; using ALL prompts")

    kept = []
    for i, row in enumerate(rows):
        if clean is not None and i not in clean:
            continue
        # Carry the ORIGINAL index so prompt_idx is comparable to every cache the
        # original project produced. It must ride inside `ground_truth`: the
        # sampler rebuilds each output row from a fixed field list and copies
        # `ground_truth` through verbatim, so a top-level key would be silently
        # dropped and every downstream index would quietly fall back to a local
        # 0..N-1 numbering.
        row = dict(row)
        gt = dict(row.get("ground_truth") or {})
        gt.setdefault("numbers", list(row["nums"]))
        gt.setdefault("target", int(row["target"]))
        gt["_orig_prompt_idx"] = i
        row["ground_truth"] = gt
        row["_orig_prompt_idx"] = i
        kept.append(row)
        if ev.get("max_prompts") and len(kept) >= int(ev["max_prompts"]):
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[harvest] prompt subset: {len(kept)} prompts -> {out_path}")
    return len(kept)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def sample_rollouts(cfg: dict, model_path: str, prompts_jsonl: Path, out_json: Path) -> bool:
    """Run the project's vLLM sampler as a subprocess. Returns True on success.

    A subprocess rather than an in-process call because vLLM does not reliably
    release GPU memory on teardown, and this loop loads a different model for
    every checkpoint.
    """
    if out_json.exists() and out_json.stat().st_size > 0:
        print(f"[harvest]   rollouts exist, skipping sampling: {out_json.name}")
        return True
    ev = cfg["eval_set"]
    cmd = [
        sys.executable, str(_SAMPLER),
        "--model_path", model_path,
        "--input_jsonl", str(prompts_jsonl),
        "--output_json", str(out_json),
        "--num_responses", str(ev.get("rollouts_per_prompt", 8)),
        "--temperature", str(ev.get("temperature", 1.0)),
        "--top_p", str(ev.get("top_p", 1.0)),
        "--top_k", str(ev.get("top_k", -1)),
        "--max_tokens", str(ev.get("max_tokens", 1024)),
        "--max_model_len", str(cfg.get("max_seq_len", 2048)),
    ]
    stops = ev.get("stop_strings") or []
    if stops:
        cmd += ["--stop_strings", ",".join(stops)]
    print(f"[harvest]   sampling: {' '.join(cmd[-14:])}")
    res = subprocess.run(cmd, cwd=str(paths.PAPER_ROOT))
    if res.returncode != 0:
        print(f"[harvest]   SAMPLING FAILED (rc={res.returncode}); snapshot left absent")
        return False
    return True


# ---------------------------------------------------------------------------
# activation extraction
# ---------------------------------------------------------------------------


def extract_activations(
    model_path: str,
    rollouts_json: Path,
    layers: list[int],
    *,
    max_seq_len: int,
    subset_map: dict[int, int] | None = None,
    max_responses_per_prompt: int | None = None,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Cache the `</think>` hidden state at each requested layer.

    NOTE on layer indexing: `hidden_states[L]` is the OUTPUT of decoder block
    L-1 (index 0 is the embedding output). The original project's probe was
    trained on `hidden_states[16]`, so that indexing is preserved verbatim here.
    Phase 2's steering hook must account for the offset — see steering.py.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("fast tokenizer required (offset_mapping)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    rows = labels_mod.load_eval_rows(rollouts_json)
    store: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    meta: list[dict] = []
    n_seen = n_kept = 0

    for local_idx, row in enumerate(rows):
        prompt = row["prompt"]
        prompt_idx = _orig_index(row, local_idx, subset_map)
        responses = row["response"]
        if max_responses_per_prompt:
            responses = responses[:max_responses_per_prompt]
        for resp_idx, resp in enumerate(responses):
            n_seen += 1
            full = prompt + resp
            close_char = full.find(THINK_CLOSE, len(prompt))
            if close_char < 0:
                continue
            enc = tok(
                full, return_offsets_mapping=True, truncation=True,
                max_length=max_seq_len, return_tensors="pt",
            )
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            tok_idx = char_to_token_index(offsets, close_char)
            if tok_idx is None:
                continue
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"].to("cuda"), output_hidden_states=True)
            for L in layers:
                store[L].append(out.hidden_states[L][0, tok_idx].float().cpu().numpy())
            meta.append({"prompt_idx": prompt_idx, "resp_idx": resp_idx, "tok_idx": tok_idx})
            n_kept += 1
        if (local_idx + 1) % 50 == 0:
            print(f"[harvest]     forward {local_idx + 1}/{len(rows)} prompts, {n_kept} rows")

    del model
    torch.cuda.empty_cache()
    print(f"[harvest]   extracted {n_kept}/{n_seen} rollouts with a locatable </think>")
    return {L: np.stack(v).astype(np.float32) for L, v in store.items() if v}, pd.DataFrame(meta)


def build_labels(rollouts_json: Path, meta: pd.DataFrame,
                 subset_map: dict[int, int] | None = None) -> pd.DataFrame:
    """Join recomputed verifier labels + structural descriptors onto cached rows.

    Delegates to `labels.build_label_table` rather than calling `block_labels` +
    `template_features` directly. The hand-rolled version omitted the PLACEBO
    columns and the sampler's own verifier scores, so harvested caches silently
    lost the control that makes the H0 residualisation readable at all — while
    ingested caches kept it. One code path, one column set.
    """
    lab = labels_mod.build_label_table(rollouts_json)
    rows = labels_mod.load_eval_rows(rollouts_json)
    # build_label_table keys on enumerate order; remap to original indices.
    remap = {i: _orig_index(row, i, subset_map) for i, row in enumerate(rows)}
    lab["prompt_idx"] = lab["prompt_idx"].map(remap)
    merged = meta.merge(lab, on=["prompt_idx", "resp_idx"], how="left", validate="one_to_one")
    if merged.isna().any().any():
        bad = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"label join left NaNs in {bad}; meta and rollouts are out of sync")
    return merged


def sanity_check_probe(frozen, X: np.ndarray, lab: pd.DataFrame, label_rule: str) -> dict:
    """Catch the historical failure mode: a mis-extracted activation saturates the probe.

    A probe scoring >0.95 on most rows is the signature of an off-by-N token
    position, not of a suspiciously good policy. Reported loudly; never fatal,
    because at a late post-Goodhart checkpoint genuine saturation is the result.
    """
    s = probes.probe_scores(frozen, X)
    y = lab[label_rule].to_numpy().astype(int)
    rep = {
        "probe_mean": float(s.mean()),
        "probe_std": float(s.std()),
        "frac_above_095": float((s > 0.95).mean()),
        "auroc": probes.auroc(y, s),
        "pos_rate": float(y.mean()),
    }
    print(
        f"[harvest]   probe sanity: mean={rep['probe_mean']:.3f} std={rep['probe_std']:.3f} "
        f"frac>0.95={rep['frac_above_095']:.3f} auroc={rep['auroc']:.3f} acc={rep['pos_rate']:.3f}"
    )
    return rep


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default=None, help="comma-separated subset of steps")
    ap.add_argument("--arms", default=None, help="comma-separated subset: on_policy,fixed_text")
    ap.add_argument("--work_dir", default="/vol/fragility", help="volume-backed scratch")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    registry.set_all_seeds(int(cfg.get("seed", 0)))

    run_id = cfg["name"]
    work = Path(args.work_dir)
    rollout_dir = work / "rollouts" / run_id
    rollout_dir.mkdir(parents=True, exist_ok=True)
    # Activations live on the volume so a later analysis job can read them.
    import os

    os.environ.setdefault("FRAGILITY_ACTS_DIR", str(work / "acts"))
    paths.mkdirs()

    prompts_jsonl = work / "prompts" / f"{run_id}.jsonl"
    n_prompts = build_prompt_subset(cfg, prompts_jsonl)
    subset_map = {
        i: int((r.get("ground_truth") or {}).get("_orig_prompt_idx", i))
        for i, r in enumerate(labels_mod.load_eval_rows(prompts_jsonl))
    }

    ladder_cfg = cfg["ladder"]
    ladder = checkpoint_logging.resolve_ladder(ladder_cfg)
    summary = checkpoint_logging.ladder_summary(ladder)
    print(f"[harvest] ladder: {summary}")

    expected = set(int(s) for s in ladder_cfg.get("expected_steps", []))
    found = {c.step for c in ladder}
    if expected - found:
        print(f"[harvest] WARNING: expected steps absent from the volume: {sorted(expected - found)}")

    # The fixed-text reference is the FULL ladder's first step, resolved before
    # any --steps filtering. Deriving it from the filtered list made the
    # reference text depend on which invocation ran, so two partial harvests
    # would write two different "fixed" texts into the same cache — destroying
    # the one property the arm exists to provide.
    reference_step = min(c.step for c in ladder)
    if args.steps:
        want = {int(s) for s in args.steps.split(",")}
        ladder = [c for c in ladder if c.step in want]
    arms = (
        args.arms.split(",")
        if args.arms
        else [a for a, on in cfg.get("arms", {}).items() if on]
    )

    frozen, prov = probes.load_frozen_probe(_resolve(cfg["probe"]["pipeline"]))
    layers = [int(L) for L in cfg["layers"]]
    label_rule = cfg.get("label_rule", "last_block")
    max_seq_len = int(cfg.get("max_seq_len", 2048))

    report: list[dict] = []
    base_rollouts: Path | None = None

    for ckpt in ladder:
        print(f"\n[harvest] === step {ckpt.step} === {ckpt.model_path}")
        rec: dict = {"step": ckpt.step, "model_path": ckpt.model_path}
        out_json = rollout_dir / f"step_{ckpt.step}.jsonl"

        if "on_policy" in arms:
            if not sample_rollouts(cfg, ckpt.model_path, prompts_jsonl, out_json):
                rec["on_policy"] = "sampling_failed"
                report.append(rec)
                continue
            if ckpt.step == reference_step:
                base_rollouts = out_json
            try:
                X_by_layer, meta = extract_activations(
                    ckpt.model_path, out_json, layers, max_seq_len=max_seq_len,
                    subset_map=subset_map,
                )
                lab = build_labels(out_json, meta, subset_map)
                probe_layer = int(cfg["probe"]["layer"])
                rec["on_policy_probe"] = sanity_check_probe(
                    frozen, X_by_layer[probe_layer], lab, label_rule
                )
                activations.save_snapshot(
                    run_id, ckpt.step, X_by_layer, lab,
                    manifest={
                        "arm": "on_policy", "model_path": ckpt.model_path,
                        "rollouts": str(out_json), "n_prompts": n_prompts,
                        "probe_provenance": prov, "sampling": cfg["eval_set"],
                    },
                    overwrite=args.overwrite,
                )
                rec["on_policy"] = "ok"
            except Exception as exc:
                print(f"[harvest]   ON-POLICY FAILED: {exc}")
                rec["on_policy"] = f"failed: {exc}"

        if "fixed_text" in arms and base_rollouts is None:
            # Reference rollouts may exist on disk from an earlier invocation.
            candidate = rollout_dir / f"step_{reference_step}.jsonl"
            if candidate.exists() and candidate.stat().st_size > 0:
                base_rollouts = candidate
                print(f"[harvest]   fixed-text reference: {candidate.name} (from an earlier run)")
            else:
                print(
                    f"[harvest]   WARNING: fixed_text requested but no reference rollouts "
                    f"at step {reference_step}; the arm is being SKIPPED. "
                    f"Harvest step {reference_step} first."
                )
                rec["fixed_text"] = f"skipped: no reference rollouts at step {reference_step}"

        if "fixed_text" in arms and base_rollouts is not None:
            try:
                X_by_layer, meta = extract_activations(
                    ckpt.model_path, base_rollouts, layers, max_seq_len=max_seq_len,
                    subset_map=subset_map,
                )
                lab = build_labels(base_rollouts, meta, subset_map)
                activations.save_snapshot(
                    f"{run_id}__fixed_text", ckpt.step, X_by_layer, lab,
                    manifest={
                        "arm": "fixed_text", "model_path": ckpt.model_path,
                        "rollouts": str(base_rollouts),
                        "note": "text held fixed at the ladder's first step; labels constant",
                    },
                    overwrite=args.overwrite,
                )
                rec["fixed_text"] = "ok"
            except Exception as exc:
                print(f"[harvest]   FIXED-TEXT FAILED: {exc}")
                rec["fixed_text"] = f"failed: {exc}"

        report.append(rec)
        _commit_volume()

    for rid in (run_id, f"{run_id}__fixed_text"):
        if activations.list_steps(rid):
            activations.save_run_manifest(
                rid,
                {"run_id": rid, "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                 "ladder": summary, "report": report},
            )
    registry.register(cfg, phase="phase0", ladder=summary, report=report)

    ok = sum(1 for r in report if r.get("on_policy") == "ok")
    print(f"\n[harvest] on-policy snapshots written: {ok}/{len(report)}")
    print(f"[harvest] acts root: {paths.acts_root()}")
    if ok < len(report):
        print("[harvest] failed checkpoints are ABSENT from the cache, not filled in.")


if __name__ == "__main__":
    main()
