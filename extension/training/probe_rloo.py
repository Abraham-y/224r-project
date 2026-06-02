"""RLOO with linear probe as reward signal.

The standard Countdown verifier scores the LAST <answer> block in a rollout
with 0/0.1/1.0. The trace-final `</think>` probe (held-out balanced AUROC
0.982 on C_outcome with corrected next-block-correctness labels) is a
near-oracle proxy for first-`<answer>`-block correctness.

This trainer replaces the verifier with the **fixed** linear probe applied to
the **current policy's** L16 hidden state at the `</think>` token.

  Variant A (default): r = probe(L16 at </think>)
  Variant B (--probe_hybrid): r = 0.5 * probe + 0.5 * verifier(last_block)

The probe is fixed throughout training. We extract hidden states from the
current policy via a transformers reference model loaded in the main process
and reloaded from the latest checkpoint each RLOO round so it tracks the
current policy. The probe-as-reward question:

  Does the policy learn to push L16 in the probe direction without genuinely
  getting more answers correct? If yes (probe score rises, verifier accuracy
  flat/drops), the linear probe direction has become a control axis of the
  policy -- Goodhart manifests as a measurable representational signature.

CLI:
  --probe_pkl   path to pickled sklearn Pipeline (StandardScaler+LogReg)
                default: extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl
  --probe_hybrid  enable variant B (probe + verifier hybrid)
  --probe_layer   L16 default
"""

from __future__ import annotations

import os
import re
import sys
import pickle


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RLOO_DIR = os.path.join(_REPO_ROOT, "rloo_trainer")
for path in (_RLOO_DIR, _REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


_THINK_CLOSE_TOKEN = "</think>"
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _pop_arg(name: str, default: str | None = None) -> str | None:
    """Pop --name VALUE from argv; return VALUE or default."""
    out = []
    found = default
    skip = False
    for i, tok in enumerate(sys.argv):
        if skip:
            skip = False
            continue
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            found = sys.argv[i + 1]
            skip = True
            continue
        out.append(tok)
    sys.argv[:] = out
    return found


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


def _install_probe_reward(probe_pkl_path: str, hybrid: bool, layer: int) -> None:
    """Replace evaluation.countdown.compute_score with a probe-based scorer.

    The new compute_score expects (solution_str, ground_truth, ...) like the
    original. We load the probe once and use a lazily-initialized reference
    model + tokenizer to extract hidden states per rollout.

    We also accumulate the probe score AND the verifier score per rollout,
    then patch wandb.log to emit train/probe_mean, train/verifier_mean,
    train/probe_minus_verifier alongside the standard train/reward_mean.
    The verifier is NOT used in training; it's only logged so we can detect
    Goodhart (probe rises while verifier stays flat/drops).
    """
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import evaluation.countdown as countdown

    original_compute_score = countdown.compute_score
    validate_equation = countdown.validate_equation
    evaluate_equation = countdown.evaluate_equation

    with open(probe_pkl_path, "rb") as f:
        probe = pickle.load(f)
    print(f"[probe_rloo] loaded probe from {probe_pkl_path}", flush=True)
    print(f"[probe_rloo] probe coef shape: {probe.named_steps['lr'].coef_.shape}", flush=True)
    print(f"[probe_rloo] hybrid mode: {hybrid}", flush=True)
    print(f"[probe_rloo] probe layer: L{layer}", flush=True)

    # The reference model is loaded lazily on first call (after the sampling
    # worker has produced rollouts, so we know the latest_checkpoint exists).
    state = {"model": None, "tokenizer": None, "checkpoint_path": None, "device": "cuda" if torch.cuda.is_available() else "cpu"}

    def _find_latest_checkpoint():
        """Find the most recently modified latest_checkpoint under any probe_rloo
        run dir (auto-detects the current run rather than hardcoding the name)."""
        import glob
        candidates = glob.glob("/vol/checkpoints/rloo_probe_checkpoints/*/*/latest_checkpoint/model")
        if not candidates:
            return None
        return max(candidates, key=lambda p: os.path.getmtime(p))

    def _ensure_reference_model():
        latest = _find_latest_checkpoint()
        if latest is None:
            return False  # no checkpoint yet (step 0 sample uses init weights)
        latest_mtime = os.path.getmtime(latest)
        if state["model"] is None or state.get("loaded_mtime", 0) < latest_mtime or state.get("loaded_from") != latest:
            print(f"[probe_rloo] (re)loading reference model from {latest} (mtime={latest_mtime})", flush=True)
            if state["model"] is not None:
                del state["model"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            state["model"] = AutoModelForCausalLM.from_pretrained(latest, torch_dtype=torch.bfloat16).to(state["device"])
            state["model"].eval()
            if state["tokenizer"] is None:
                state["tokenizer"] = AutoTokenizer.from_pretrained(latest)
            state["loaded_mtime"] = latest_mtime
            state["loaded_from"] = latest
        return True

    def _init_reference_from(model_path: str):
        """One-time init: load reference model from a known path (e.g. C_SFT) for step 0."""
        if state["model"] is not None:
            return
        print(f"[probe_rloo] initial reference model load from {model_path}", flush=True)
        state["model"] = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(state["device"])
        state["model"].eval()
        state["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        state["loaded_mtime"] = 0.0  # forces reload from disk once latest_checkpoint appears

    # Initial reference model: C_outcome (where the probe was trained).
    # The probe sees its training distribution at step 0; as the policy
    # updates each round, we reload from the latest_checkpoint.
    init_path = os.environ.get(
        "PROBE_RLOO_INIT_MODEL",
        "/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model",
    )
    try:
        _init_reference_from(init_path)
    except Exception as e:
        print(f"[probe_rloo] WARNING: could not load initial reference model from {init_path}: {e}", flush=True)

    # Reconstruct the asingh15-style chat-template prompt from ground_truth so
    # the probe sees hidden states with the SAME context it was trained on
    # (prompt + response tokenized together). Without this, the probe receives
    # OOD activations and saturates to ~0.98 for every rollout (we hit this
    # in run4 — reward_mean = 0.98 at step 0).
    _PROMPT_TEMPLATE = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nA conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides "
        "the user with the answer.\nUser: Using the numbers [{nums}], create an "
        "equation that equals {target}. You can use basic arithmetic operations "
        "(+, -, *, /) and each number can only be used once. Show your work in "
        "<think> </think> tags. And return the final answer in <answer> </answer> "
        "tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me "
        "solve this step by step.<|im_end|>\n<|im_start|>assistant\n"
    )

    def _reconstruct_prompt(ground_truth) -> str:
        try:
            import numpy as _np
            nums_str = _np.array2string(_np.asarray(list(ground_truth["numbers"])),
                                         separator=" ").strip("[]").strip()
            return _PROMPT_TEMPLATE.format(nums=nums_str, target=ground_truth["target"])
        except Exception:
            return ""

    @torch.no_grad()
    def _probe_score(solution_str: str, ground_truth=None) -> float:
        """Return scalar probe probability in [0, 1] for the rollout's </think> position.

        IMPORTANT: tokenizes prompt+response together (matching how the probe
        was trained in cache_hidden_states.py), then locates </think> AFTER the
        prompt. Without the prompt context, the probe receives OOD activations
        and saturates.

        Returns 0.0 if </think> token not found, model not loaded, or any error.
        """
        if state["model"] is None or state["tokenizer"] is None:
            return 0.0
        if _THINK_CLOSE_TOKEN not in solution_str:
            return 0.0
        # Build prompt + response prefix-up-to-</think>
        idx = solution_str.find(_THINK_CLOSE_TOKEN) + len(_THINK_CLOSE_TOKEN)
        response_prefix = solution_str[:idx]
        prompt_text = _reconstruct_prompt(ground_truth) if ground_truth is not None else ""
        full_text = prompt_text + response_prefix
        try:
            input_ids = state["tokenizer"](full_text, return_tensors="pt").input_ids.to(state["device"])
            if input_ids.shape[1] < 2: return 0.0
            out = state["model"](input_ids, output_hidden_states=True, use_cache=False)
            # Last token corresponds to the final subtoken of the response's </think>
            h = out.hidden_states[layer][0, -1, :].float().cpu().numpy().reshape(1, -1)
            score = float(probe.predict_proba(h)[0, 1])
            return score
        except Exception as e:
            print(f"[probe_rloo] _probe_score error: {e}", flush=True)
            return 0.0

    # Tracker for dual logging (probe vs verifier) — used by patched wandb.log
    _TRACKER = {"probe": [], "verifier": []}

    def probe_compute_score(solution_str, ground_truth, method="strict",
                            format_score=0.1, score=1.0):
        """Probe-based reward.

        Variant A (default): r = probe_score
        Variant B (hybrid): r = 0.5 * probe_score + 0.5 * original_verifier_score

        Logs BOTH probe_score and verifier_score per rollout to a shared
        tracker so we can detect Goodhart in wandb (probe rises while
        verifier doesn't).
        """
        _ensure_reference_model()
        probe_s = _probe_score(solution_str, ground_truth)
        verifier_s = original_compute_score(solution_str, ground_truth, method, format_score, score)
        _TRACKER["probe"].append(probe_s)
        _TRACKER["verifier"].append(verifier_s)
        if not hybrid:
            return probe_s
        return 0.5 * probe_s + 0.5 * verifier_s

    probe_compute_score.__name__ = "compute_score"
    probe_compute_score.__wrapped__ = original_compute_score
    countdown.compute_score = probe_compute_score
    print("[probe_rloo] PROBE REWARD active: r = " + ("0.5*probe + 0.5*verifier" if hybrid else "probe (Variant A)"), flush=True)

    # Patch the Run.log method (rloo.py uses self.wandb.log() on the Run object,
    # NOT the module-level wandb.log()) so our dual-logging actually fires.
    try:
        import wandb
        from wandb.sdk.wandb_run import Run as _WandbRun
        _orig_log = _WandbRun.log

        def _patched_log(self, data=None, *args, **kwargs):
            try:
                if isinstance(data, dict) and "train/reward_mean" in data and _TRACKER["probe"]:
                    p = float(np.mean(_TRACKER["probe"]))
                    v = float(np.mean(_TRACKER["verifier"]))
                    data = dict(data)
                    data["train/probe_mean"] = p
                    data["train/verifier_mean"] = v
                    data["train/probe_minus_verifier"] = p - v
                    data["train/probe_count"] = len(_TRACKER["probe"])
                    print(f"[probe_rloo] step probe_mean={p:.4f} verifier_mean={v:.4f} gap={p-v:+.4f} n={len(_TRACKER['probe'])}", flush=True)
                    _TRACKER["probe"].clear()
                    _TRACKER["verifier"].clear()
            except Exception as e:
                print(f"[probe_rloo] dual-log patch warning: {e}", flush=True)
            return _orig_log(self, data, *args, **kwargs)

        _WandbRun.log = _patched_log
        print("[probe_rloo] dual-logging active: patched wandb.sdk.wandb_run.Run.log", flush=True)
    except Exception as e:
        print(f"[probe_rloo] WARNING: could not patch Run.log for dual logging: {e}", flush=True)


def main() -> None:
    hybrid = _pop_switch("probe_hybrid")
    probe_pkl = _pop_arg("probe_pkl", "/vol/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl")
    layer = int(_pop_arg("probe_layer", "16"))
    _install_probe_reward(probe_pkl, hybrid, layer)

    rloo_path = os.path.join(_RLOO_DIR, "rloo.py")
    sys.argv[0] = rloo_path
    with open(rloo_path) as f:
        code = compile(f.read(), rloo_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": rloo_path})


if __name__ == "__main__":
    main()
