"""Phase 2 — activation patching between correct and incorrect runs.

READ phase2_predictions.md FIRST.

Protocol, per checkpoint and per layer:
  1. pick pairs of rollouts ON THE SAME PROMPT, one verifier-correct and one
     verifier-wrong, both with a locatable `</think>`
  2. take the donor's `</think>` activation at layer L
  3. run the recipient's prefix, REPLACING its `</think>` activation with the
     donor's, and regenerate
  4. record whether correctness flipped

Pairs are same-prompt by construction. A cross-prompt donor carries the wrong
problem's numbers, so a flip would measure "we destroyed the computation", not
"we transplanted a correctness state".

Patching at bracketing layers as well as the probe layer is what detects
relocation: under H1 the probe layer should lose patching efficacy while some
other layer gains it.

Metrics are flushed (and the Modal volume committed) after every checkpoint, not
once at the end: a job killed by the Modal timeout otherwise loses every
checkpoint it had already finished.

    modal run --detach modal_fragility.py patching -- \
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


def build_pairs(rollouts_json: Path, n_pairs: int, seed: int) -> list[dict]:
    """Same-prompt (correct, wrong) rollout pairs, deterministically chosen."""
    rows = labels_mod.load_eval_rows(rollouts_json)
    rng = np.random.RandomState(seed)
    pairs: list[dict] = []
    order = rng.permutation(len(rows))
    for local_idx in order:
        row = rows[int(local_idx)]
        target, nums = int(row["target"]), list(row["nums"])
        correct, wrong = [], []
        for r_idx, resp in enumerate(row["response"]):
            if THINK_CLOSE not in resp:
                continue
            lab = labels_mod.block_labels(resp, target, nums)
            (correct if lab["last_block"] == 1 else wrong).append((r_idx, resp))
        if not correct or not wrong:
            continue
        c_i, c_txt = correct[rng.randint(len(correct))]
        w_i, w_txt = wrong[rng.randint(len(wrong))]
        pairs.append(
            {
                "prompt": row["prompt"], "target": target, "nums": nums,
                # ORIGINAL 0..499 index (the sampler drops top-level keys, so it
                # rides inside `ground_truth`), raising rather than falling back
                # to the local position — see labels.orig_prompt_idx.
                "prompt_idx": labels_mod.orig_prompt_idx(row, int(local_idx)),
                "correct_idx": c_i, "correct_text": c_txt,
                "wrong_idx": w_i, "wrong_text": w_txt,
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


def _prefix_and_pos(tok, prompt: str, response: str, max_seq_len: int):
    """Prefix ending at `</think>`, and the token index the PROBE reads.

    `</think>` is three tokens ('</', 'think', '>'), so the prefix's last token
    sits two positions past the probe's read site and carries a different token
    identity than the calibration sequence ('>' vs '>\\n\\n'). Patching there
    transplants a state into a slot the probe never looks at. Same rule as
    cache_hidden_states, via the same imported helper.
    """
    full = prompt + response
    close = full.find(THINK_CLOSE, len(prompt))
    if close < 0:
        return None, None, None
    prefix = full[: close + len(THINK_CLOSE)]
    enc = tok(prefix, return_offsets_mapping=True, truncation=True,
              max_length=max_seq_len, return_tensors="pt")
    offs = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
    pos = char_to_token_index(offs, close)
    if pos is None:
        return None, None, None  # truncated away; skip rather than guess
    return prefix, enc["input_ids"], pos


def run_checkpoint(
    *, model_path: str, step: int, pairs: list[dict], layers: list[int],
    max_seq_len: int, max_new_tokens: int, seed: int = 0,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluation.countdown import compute_score

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    rows: list[dict] = []
    donor_cache: dict[tuple[int, str], dict[int, Any]] = {}
    for layer in layers:
        if not 1 <= layer < model.config.num_hidden_layers:
            # hidden_states[0] is the embedding output and has no decoder layer
            # to hook; hidden_states[n_layers] is post-final-layernorm while a
            # layer hook sees pre-norm. Clamping either into range would patch a
            # vector from one space into another and report it as a result.
            raise ValueError(
                f"layer {layer} is not patchable: expected 1 <= layer < "
                f"{model.config.num_hidden_layers}"
            )
        # hook layer L-1 so its OUTPUT is hidden_states[L], the probe's basis.
        hook_layer = layer - 1
        with steering.intervention(model, hook_layer) as state:
            for p_i, pair in enumerate(pairs):
                gt = {"target": pair["target"], "numbers": pair["nums"]}
                # Both directions: correct donor -> wrong recipient, and reverse.
                for donor_key, recip_key, donor_label in (
                    ("correct_text", "wrong_text", 1),
                    ("wrong_text", "correct_text", 0),
                ):
                    _dp, d_ids, d_pos = _prefix_and_pos(
                        tok, pair["prompt"], pair[donor_key], max_seq_len
                    )
                    _rp, r_ids, r_pos = _prefix_and_pos(
                        tok, pair["prompt"], pair[recip_key], max_seq_len
                    )
                    if d_ids is None or r_ids is None:
                        continue

                    # One forward returns every layer, so cache it per
                    # (pair, direction) instead of recomputing for each layer.
                    ck = (p_i, donor_key)
                    if ck not in donor_cache:
                        state.mode = "off"
                        with torch.no_grad():
                            out = model(
                                input_ids=d_ids.to("cuda"), output_hidden_states=True
                            )
                        donor_cache[ck] = {
                            L: out.hidden_states[L][0, d_pos].detach().clone()
                            for L in layers
                        }
                    donor_vec = donor_cache[ck][layer]

                    # PAIRED design: the same prefix is regenerated twice under
                    # the same sampling seed, once unpatched and once patched.
                    # The control absorbs the resampling churn that would
                    # otherwise be indistinguishable from a patch effect.
                    outcomes = {}
                    for cond in ("control", "patch"):
                        if cond == "control":
                            state.mode, state.vector, state.position = "off", None, None
                        else:
                            state.mode, state.vector, state.position = (
                                "patch", donor_vec, r_pos,
                            )
                        state.reset()
                        torch.manual_seed(seed * 100003 + p_i * 17 + layer)
                        with torch.no_grad():
                            gen = model.generate(
                                input_ids=r_ids.to("cuda"),
                                max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                pad_token_id=tok.eos_token_id,
                            )
                        if cond == "patch" and not state.applied:
                            raise RuntimeError(f"patch hook never fired (step {step}, L{layer})")
                        text = tok.decode(gen[0][r_ids.shape[1]:], skip_special_tokens=False)
                        outcomes[cond] = float(compute_score(text, gt) == 1.0)

                    rows.append(
                        {
                            "checkpoint_step": step, "layer": layer,
                            "prompt_idx": pair["prompt_idx"],
                            "donor_label": donor_label,
                            # recipient's original label is the complement
                            "orig_correct": float(1 - donor_label),
                            "new_correct": outcomes["patch"],
                            "control_correct": outcomes["control"],
                        }
                    )
                if (p_i + 1) % 25 == 0:
                    print(f"[patch]   step {step} L{layer}: {p_i + 1}/{len(pairs)} pairs")

    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--work_dir", default="/vol/fragility")
    ap.add_argument("--steps", default=None)
    args = ap.parse_args()

    cfg = registry.load_config(args.config)
    seed = int(cfg.get("seed", 42))
    registry.set_all_seeds(seed)

    entry = next((l for l in cfg["ladders"] if l["name"] == args.ladder), None)
    if entry is None:
        raise SystemExit(f"--ladder {args.ladder!r} not in config")
    lcfg = registry.load_config(entry["config"])
    run_id = lcfg["name"]

    pcfg = cfg.get("patching", {})
    if not pcfg.get("enabled", True):
        print("[patch] patching disabled in config; nothing to do")
        return

    work_dir = Path(args.work_dir)
    rollouts = work_dir / "rollouts" / run_id
    steps = [int(s) for s in (args.steps.split(",") if args.steps else cfg["checkpoint_steps"])]
    ladder = {c.step: c for c in checkpoint_logging.resolve_ladder(lcfg["ladder"])}
    missing = [s for s in steps if s not in ladder]
    if missing:
        raise SystemExit(
            f"steps {missing} not in the ladder (available: {sorted(ladder)}). "
            "Check `ladder.include_final` / `final_step` in the harvest config."
        )

    writer = metrics_io.MetricWriter(run_id, "phase2", seed)
    raw_dir = work_dir / "patching" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    def flush(step: int) -> None:
        """Persist this checkpoint's metrics NOW.

        One flush at the end of the job means a Modal timeout on the last of five
        checkpoints discards all five — the metrics only exist in memory until
        then. Shards are separate files, so flushing per checkpoint costs
        nothing, and `metrics_io.latest` deduplicates by write time. The volume
        commit is the other half: the launcher commits in a `finally`, which a
        hard timeout skips (harvest_ladder commits per checkpoint for exactly
        this reason).
        """
        shard = writer.flush()
        if shard is not None:
            print(f"[patch]   metrics -> {shard}")
            paths.commit_volume("[patch]   ")

    for step in steps:
        rj = rollouts / f"step_{step}.jsonl"
        if step not in ladder or not rj.exists():
            print(f"[patch] step {step}: checkpoint or rollouts absent; no rows written")
            continue
        pairs = build_pairs(rj, int(pcfg.get("n_pairs", 150)), seed)
        if not pairs:
            # Late post-Goodhart checkpoints may have almost no correct rollouts.
            print(f"[patch] step {step}: no same-prompt correct/wrong pairs available")
            writer.add("patch_n_pairs", 0.0, checkpoint_step=step, arm="patching")
            flush(step)
            continue
        print(f"[patch] step {step}: {len(pairs)} same-prompt pairs")

        rows = run_checkpoint(
            model_path=ladder[step].model_path, step=step, pairs=pairs,
            layers=[int(L) for L in pcfg.get("layers", [16])],
            max_seq_len=int(lcfg.get("max_seq_len", 2048)),
            max_new_tokens=int(pcfg.get("max_new_tokens", 400)),
            seed=seed,
        )
        # Temp file + rename, so a non-empty raw file is always a complete
        # checkpoint rather than a truncated one from an interrupted write.
        out = raw_dir / f"step_{step}.jsonl"
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, out)

        import pandas as pd

        df = pd.DataFrame(rows)
        for layer, grp in df.groupby("layer"):
            agg = steering.flip_rate(grp.to_dict("records"))
            agg["patch_n_pairs"] = float(len(pairs))
            for k, v in agg.items():
                writer.add(k, v, checkpoint_step=step, layer=int(layer), arm="patching")
            print(f"[patch]   step {step} L{layer}: {agg}")
        flush(step)

    # Usually a no-op: every checkpoint already flushed. Kept so any row added
    # outside the loop still lands.
    p = writer.flush()
    registry.register(cfg, phase="phase2", ladder=args.ladder, kind="patching", steps=steps)
    print(f"[patch] done; final shard: {p}" if p else "[patch] done (all shards flushed per step)")
    paths.commit_volume("[patch] ")


if __name__ == "__main__":
    main()
