# Extension: Concealment Under Outcome Pressure

Companion code for `extension.md`. This is the AI-assistable extension territory; the default-project trainers (`sft_trainer/`, `ipo_trainer/`, `rloo_trainer/`) stay untouched.

## Architecture

```
extension/
├── README.md              # this file
├── metrics/               # behavioral + calibration + dynamics (Person B)
│   ├── behavioral.py      # Layer A metrics over an eval JSON
│   ├── calibration.py     # ECE + reliability diagrams
│   └── dynamics.py        # Layer C: replay metrics over the 10-step C_outcome snapshots
├── subgoal/               # subgoal grammar + reward (Person A)
│   ├── parser.py          # <subgoal>...</subgoal> -> structured triples
│   ├── verifier.py        # exact is_reachable / is_achieved checks
│   ├── reward.py          # R = R_outcome + λ * R_subgoal
│   └── sft_augment.py     # insert subgoals into the warm-start traces
└── probe/                 # hidden-state probes (Person B)
    ├── cache_hidden_states.py  # extract activations at (layer, position) -> .npz
    ├── train_probe.py          # logistic regression per (ckpt, layer, position)
    ├── eval_probe.py           # concealment gap, earliness, Cohen's d
    └── transfer.py             # 3x3 cross-checkpoint AUROC matrix
```

## Division of labor

| Person | Owns | Compute |
|---|---|---|
| **A** (RL + subgoal infra) | `subgoal/*`, `C_SFT_aug` + `C_process` training, dynamics replay over saved snapshots | Modal GPU for the two training runs |
| **B** (metrics + probe analysis) | `metrics/*`, `probe/*` | Laptop for Layer A and probe training; cheap Modal use for hidden-state extraction |

The intersections are small and explicit:
- Person A's `subgoal/verifier.py` is imported by Person B's `metrics/behavioral.py` *only* once `C_process` exists (used to score subgoal-level metrics). Until then, Person B can ignore it.
- Person B's `probe/cache_hidden_states.py` needs the checkpoint paths Person A produces (`C_SFT_aug`, `C_process`). Until those exist, Person B works against `C_SFT` (asingh15) and `C_outcome` (on the Modal volume).

## How to run the Day-1 deliverables (no GPU required)

Both partners can do these on their laptop with only `numpy + matplotlib` installed.

### Person B — Layer A behavioral metrics
The two eval JSONs are produced by `evaluation/countdown_eval.py` and live at the repo root:

```bash
# from the project root
python extension/metrics/behavioral.py \
  --sft_json eval_sft.json --rloo_json eval_rloo.json \
  --out_dir extension/outputs/phase1
```

Prints a side-by-side comparison table and writes `phase1_metrics.csv`. This is the Phase-1 result for the report.

### Person A — subgoal parser + verifier
Run the unit tests:
```bash
python -m unittest extension/subgoal/test_parser.py extension/subgoal/test_verifier.py
```

(If `unittest` discovery doesn't pick them up, run them directly with `python extension/subgoal/parser.py` / `verifier.py` — each file has a `__main__` smoke test.)

## Checkpoints currently in use

| Name | Where | Status |
|---|---|---|
| `C_SFT` | `asingh15/qwen-sft-countdown-defaultproj` (HF) | Loaded directly from HF |
| `C_outcome` | `/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model` (Modal volume `default-proj-training`) | Done |
| `C_outcome` snapshots | `epoch_0_step_{0,10,20,…,90}/model` under the same dir | Done |
| `C_SFT_aug` | (to train) | Person A, Day 3-4 |
| `C_process` | (to train) | Person A, Day 4-6 |

## Eval JSON schema (what `metrics/` consumes)

Each line of `eval_*.json` is one prompt's record:

```json
{
  "prompt": "<chat-templated prompt>",
  "target": 28,
  "nums": [95, 11, 56],
  "ground_truth": {"target": 28, "numbers": [95, 11, 56]},
  "response": ["<rollout 1>", ..., "<rollout 16>"],
  "scores":   [0.1,           ..., 1.0           ]
}
```

`scores` come from `evaluation/countdown.py`: 0.0 (no parseable answer), 0.1 (format only / invalid equation), 1.0 (valid + correct).

## Conventions for adding new metrics

1. Keep modules **pure** when possible — take JSON paths or list of rollouts, return a dict of named floats. No global state.
2. Every module has an `if __name__ == "__main__":` smoke test that runs on the existing JSONs.
3. CSV outputs go to `extension/outputs/<phase>/` (gitignored). Plots go alongside.
4. New high-confidence regex patterns / token sets go in `metrics/behavioral.py:CONFIDENT_ASSERTIONS` so they're discoverable.
