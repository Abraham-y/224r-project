"""RLOO with PROBE-as-reward (reward-design ablation, arm C).

Reward shaping ablation:
  - arm A (outcome) : vanilla `rloo_trainer/rloo.py` (verifier 0/0.1/1.0, last block)
  - arm B (first)   : `extension/training/firstanswer_rloo.py` (verifier, first block)
  - arm C (probe)   : THIS script — reward = P(correct) from a frozen linear probe
                      read at the `</think>` token (layer L) of a FROZEN C_SFT model.

The question. The trace-final probe is a near-oracle *reader* of correctness
(AUROC ~0.98) but causal steering along it was NULL (it's a reader, not a
controller). Does it survive being made a *controller* via RL? Either outcome
is a finding:
  - works  -> "a model's own internal verifier can serve as an RL reward."
  - Goodhart-> the policy learns to inflate the probe while staying wrong
              (reward hacking of a learned reward). We log both the probe
              reward (what RL optimizes) and can re-score the resulting
              checkpoint with the real verifier downstream to detect the gap.

The reward function is STATIONARY: a frozen C_SFT model + a fixed probe form a
fixed text->scalar reward model. As the policy drifts, its rollouts are fed
through this fixed reward model.

Architecture (mirrors firstanswer_rloo.py — patch, then exec rloo.py unchanged):
  1. Load a FROZEN reward model (default: C_SFT = asingh15 SFT).
  2. Train (or load) the `</think>` L16 probe as Pipeline(StandardScaler, LR)
     from existing C_SFT rollouts (eval_c_sft_n500.json), labeled by
     first-`<answer>`-block correctness. [one-time, ~2-3 min on GPU]
  3. Monkey-patch evaluation.countdown.compute_score -> probe reward.
  4. Monkey-patch RLOODataset.collate_fn to inject the prompt into ground_truth
     (the reward fn needs prompt+response for a valid </think> hidden state;
     rloo.py's reward loop only passes the response).
  5. exec rloo_trainer/rloo.py.

GPU note. During reward computation the vLLM sampling worker is still resident,
so this wrapper leaves headroom by defaulting --gpu_memory_utilization to 0.75
(override by passing it explicitly).

CLI (wrapper flags are consumed here; everything else passes through to rloo.py):
  --probe PATH            precomputed probe (.pkl Pipeline/LR or .npz direction);
                          if omitted, the probe is trained at startup
  --probe_model PATH      frozen model for hidden states (default C_SFT)
  --probe_layer N         hidden layer index (default 16)
  --reward_mode MODE      probe | probe_gated | blend   (default probe)
  --blend_outcome W       weight on verifier reward when reward_mode=blend (default 0.3)
  --train_rollouts PATH   C_SFT rollouts to train the probe on (default eval_c_sft_n500.json)
  --probe_train_max N     cap on rollouts used to train the probe (default 3000)
  --reward_disable        revert to vanilla verifier reward (A/B control)

Example:
  modal run modal_train.py probe_reward_rloo -- \
      --wandb_project rloo_reward_ablation --wandb_name probe_reward_v1 \
      --num_training_steps 100 --save_every_n_steps 10
"""

from __future__ import annotations

import json
import os
import re
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RLOO_DIR = os.path.join(_REPO_ROOT, "rloo_trainer")
_DATA_DIR = os.path.join(_REPO_ROOT, "extension", "data")
for path in (_RLOO_DIR, _REPO_ROOT, _DATA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# argv helpers (consume wrapper-only flags before rloo.py's argparse sees them)
# ---------------------------------------------------------------------------


def _pop_value(name: str) -> str | None:
    out: list[str] = []
    val: str | None = None
    argv = sys.argv
    i = 0
    while i < len(argv):
        if argv[i] == f"--{name}":
            if i + 1 < len(argv):
                val = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        out.append(argv[i])
        i += 1
    sys.argv[:] = out
    return val


def _pop_switch(name: str) -> bool:
    present = False
    out = []
    for tok in sys.argv:
        if tok == f"--{name}":
            present = True
            continue
        out.append(tok)
    sys.argv[:] = out
    return present


# ---------------------------------------------------------------------------
# task helpers
# ---------------------------------------------------------------------------


def _first_block_correct(solution_str: str, gt: dict, countdown) -> int:
    """1 iff the FIRST <answer> block is a valid equation hitting the target."""
    m = _ANSWER_RE.search(solution_str)
    if not m:
        return 0
    eq = m.group(1).strip()
    if not countdown.validate_equation(eq, gt["numbers"]):
        return 0
    try:
        r = countdown.evaluate_equation(eq)
        return 1 if (r is not None and abs(r - gt["target"]) < 1e-5) else 0
    except Exception:
        return 0


def _reconstruct_prompt(gt: dict) -> str | None:
    """Fallback when collate didn't inject the prompt: rebuild it from numbers/target."""
    try:
        from generate_countdown import format_prompt
        return format_prompt(list(gt["numbers"]), int(gt["target"]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# frozen reward model + </think> hidden-state extraction
# ---------------------------------------------------------------------------


def _load_frozen_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[probe_reward] loading frozen reward model {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("Fast tokenizer required (need offset_mapping).")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


def _think_close_hidden(tok, model, prompt: str, response: str, layer: int,
                        max_seq_len: int = 2048):
    """Layer-`layer` hidden state at the response's first `</think>` token of
    (prompt + response), or None if there is no locatable `</think>`."""
    import torch
    full = prompt + response
    close_char = full.find(THINK_CLOSE, len(prompt))
    if close_char < 0:
        return None
    enc = tok(full, return_offsets_mapping=True, truncation=True,
              max_length=max_seq_len, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
    tok_idx = None
    for i, (s, e) in enumerate(offsets):
        if s <= close_char < e or s == close_char:
            tok_idx = i
            break
    if tok_idx is None:
        for i, (s, _e) in enumerate(offsets):
            if s >= close_char:
                tok_idx = i
                break
    if tok_idx is None or tok_idx >= input_ids.shape[1]:
        return None
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[layer][0, tok_idx].float().cpu().numpy()


# ---------------------------------------------------------------------------
# probe: train at startup (default) or load
# ---------------------------------------------------------------------------


def _train_probe(tok, model, rollouts_path: str, layer: int, max_rollouts: int,
                 countdown):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    if not os.path.exists(rollouts_path):
        raise FileNotFoundError(
            f"--train_rollouts {rollouts_path} not found; provide --probe instead.")
    rows = [json.loads(l) for l in open(rollouts_path) if l.strip()]
    print(f"[probe_reward] training </think> L{layer} probe from {rollouts_path} "
          f"({len(rows)} problems, cap {max_rollouts} rollouts)", flush=True)

    X, y, g = [], [], []
    n = 0
    for p_idx, row in enumerate(rows):
        if n >= max_rollouts:
            break
        prompt = row["prompt"]
        gt = {"target": int(row["target"]), "numbers": list(row["nums"])}
        for resp in row["response"]:
            if n >= max_rollouts:
                break
            h = _think_close_hidden(tok, model, prompt, resp, layer)
            if h is None:
                continue
            X.append(h)
            y.append(_first_block_correct(resp, gt, countdown))
            g.append(p_idx)
            n += 1
        if (p_idx + 1) % 50 == 0:
            print(f"[probe_reward]   {p_idx + 1} problems, {n} usable rollouts", flush=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    g = np.asarray(g)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"[probe_reward] {len(y)} usable rollouts; pos={n_pos} neg={n_neg}", flush=True)
    if n_pos < 10 or n_neg < 10:
        raise RuntimeError(f"Too few of one class (pos={n_pos}, neg={n_neg}).")

    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, nb, replace=False),
                          rng.choice(neg, nb, replace=False)])
    Xs, ys, gs = X[idx], y[idx], g[idx]

    def make_pipe():
        return Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(C=0.1, max_iter=2000))])

    nsplits = min(5, len(np.unique(gs)))
    if nsplits >= 2:
        preds = np.full(len(ys), np.nan)
        for tr, te in GroupKFold(nsplits).split(Xs, ys, gs):
            preds[te] = make_pipe().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
        m = ~np.isnan(preds)
        auroc = roc_auc_score(ys[m], preds[m]) if len(set(ys[m].tolist())) == 2 else float("nan")
        print(f"[probe_reward] held-out (by problem) probe AUROC = {auroc:.3f} "
              f"(sanity; expect ~0.9)", flush=True)

    pipe = make_pipe().fit(Xs, ys)
    return pipe


def _load_probe(path: str):
    import numpy as np
    if path.endswith(".npz"):
        with np.load(path) as d:
            v = d["v_input_raw"] if "v_input_raw" in d.files else d["v_unit"]
        v = np.asarray(v, dtype=np.float32)

        class _DirProbe:  # uncalibrated but monotonic; fine for within-group RLOO
            def predict_proba(self, X):
                z = np.asarray(X, dtype=np.float32) @ v
                p = 1.0 / (1.0 + np.exp(-z))
                return np.stack([1 - p, p], axis=1)
        return _DirProbe()

    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# reward fn + dataset patch
# ---------------------------------------------------------------------------


def _make_probe_reward(tok, model, probe, layer, mode, blend, orig_compute_score):
    def probe_compute_score(solution_str, ground_truth, method="strict",
                            format_score=0.1, score=1.0):
        prompt = ground_truth.get("_prompt") if isinstance(ground_truth, dict) else None
        if prompt is None:
            prompt = _reconstruct_prompt(ground_truth) or ""
        h = _think_close_hidden(tok, model, prompt, solution_str, layer)
        if h is None:
            return 0.0  # no </think> -> no readable internal state -> no reward
        import numpy as np
        p = float(probe.predict_proba(h[None, :])[0, 1])
        if mode == "probe":
            return p
        if mode == "probe_gated":
            return p if _ANSWER_RE.search(solution_str) else 0.0
        if mode == "blend":
            v = float(orig_compute_score(solution_str, ground_truth))
            return (1.0 - blend) * p + blend * v
        return p

    probe_compute_score.__name__ = "compute_score"
    probe_compute_score.__wrapped__ = orig_compute_score
    return probe_compute_score


def _patch_collate_inject_prompt():
    import rloo_trainer.rloo_dataset as rds
    orig = rds.RLOODataset.collate_fn

    def collate(self, batch):
        out = orig(self, batch)
        new_gts = []
        for gt, p in zip(out["ground_truth"], out["prompt"]):
            gt2 = dict(gt) if isinstance(gt, dict) else gt
            if isinstance(gt2, dict):
                gt2["_prompt"] = p
            new_gts.append(gt2)
        out["ground_truth"] = new_gts
        return out

    rds.RLOODataset.collate_fn = collate


# ---------------------------------------------------------------------------


def main() -> None:
    if _pop_switch("reward_disable"):
        print("[probe_reward] --reward_disable: vanilla verifier reward.", flush=True)
        _exec_rloo()
        return

    probe_path = _pop_value("probe")
    probe_model = _pop_value("probe_model") or "asingh15/qwen-sft-countdown-defaultproj"
    layer = int(_pop_value("probe_layer") or 16)
    mode = _pop_value("reward_mode") or "probe"
    blend = float(_pop_value("blend_outcome") or 0.3)
    train_rollouts = _pop_value("train_rollouts") or "eval_c_sft_n500.json"
    probe_train_max = int(_pop_value("probe_train_max") or 3000)

    # Leave GPU headroom for the frozen reward model (vLLM is resident at reward time).
    if "--gpu_memory_utilization" not in sys.argv:
        sys.argv += ["--gpu_memory_utilization", "0.75"]

    import evaluation.countdown as countdown
    orig_compute_score = countdown.compute_score

    tok, model = _load_frozen_model(probe_model)
    probe = _load_probe(probe_path) if probe_path else _train_probe(
        tok, model, train_rollouts, layer, probe_train_max, countdown)

    countdown.compute_score = _make_probe_reward(
        tok, model, probe, layer, mode, blend, orig_compute_score)
    _patch_collate_inject_prompt()
    print(f"[probe_reward] ACTIVE: reward=probe P(correct) @ </think> L{layer}, "
          f"mode={mode}, frozen model={probe_model}. RL now optimizes the probe.",
          flush=True)

    _exec_rloo()


def _exec_rloo() -> None:
    rloo_path = os.path.join(_RLOO_DIR, "rloo.py")
    sys.argv[0] = rloo_path
    with open(rloo_path) as f:
        code = compile(f.read(), rloo_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": rloo_path})


if __name__ == "__main__":
    main()
