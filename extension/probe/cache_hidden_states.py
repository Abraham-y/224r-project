"""Extract Qwen2.5-0.5B hidden states at specified (layer, position) pairs.

Person B. Day 2-3. Runs on Modal (cheap, ~$10).

Pseudocode:

    model = AutoModelForCausalLM.from_pretrained(ckpt, output_hidden_states=True)
    for row in eval_rows:                       # 50 prompts x 16 rollouts each
        for response in row["response"]:
            ids = tokenizer(row.prompt + response).input_ids
            h = model(ids).hidden_states          # tuple of (1, T, 896) per layer
            # take hidden states at requested positions:
            #   "early_prefix"  -> first 16 tokens of <think> body
            #   "mid_=" tokens  -> sampled positions of '=' token
            #   "pre_answer"    -> the </think> token position
            cache[(ckpt, layer, position)].append((h[layer, pos], score == 1.0))
    np.savez(out_path, **cache)

Outputs ``<ckpt_name>_l<layer>_<position>.npz`` with arrays X (N, 896) and y (N,).

TODO:
    * Implement above. Use bf16 forward + cast to float32 only for the cached
      activation slice (storage stays cheap).
    * For C_outcome and C_SFT_aug, load from the Modal volume path.
    * For C_SFT, load asingh15/qwen-sft-countdown-defaultproj from HF.
    * Target cache size: ~5k examples per checkpoint per (layer, position).
    * Layers to sweep: L12, L16, L20 (per §4.1).
    * Positions: early_prefix (first 16 tokens of <think> body), mid_= (sampled),
      pre_answer (</think>).
"""

from __future__ import annotations

POSITIONS = ("early_prefix", "mid_eq", "pre_answer")
DEFAULT_LAYERS = (12, 16, 20)
