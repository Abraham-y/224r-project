# PowerShell version of the clean-replication runbook commands.
# Run from repo root. See RUNBOOK_CLEAN_REPLICATION.md for the bash versions
# and full explanation. Uses dot-source pattern so .env vars are loaded once
# and shared across the steps in the same shell session.
#
# Usage examples:
#   .\scripts\runbook.ps1 -Step 2      # regenerate clean rollouts
#   .\scripts\runbook.ps1 -Step 4b     # launch probe_topk full RLOO
#   .\scripts\runbook.ps1 -Step 5      # eval probe_topk checkpoint
#
# All step numbers match RUNBOOK_CLEAN_REPLICATION.md.

param(
    [Parameter(Mandatory=$true)]
    [string]$Step
)

$ErrorActionPreference = "Stop"

# Load .env if present.
if (Test-Path ".env") {
    Write-Host "Loading .env"
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $idx = $line.IndexOf("=")
            if ($idx -gt 0) {
                $key = $line.Substring(0, $idx).Trim()
                $val = $line.Substring($idx + 1).Trim()
                if ($val.StartsWith('"') -and $val.EndsWith('"')) {
                    $val = $val.Substring(1, $val.Length - 2)
                }
                [System.Environment]::SetEnvironmentVariable($key, $val)
            }
        }
    }
}

switch ($Step) {
    "2" {
        # Step 2: regenerate clean rollouts on both checkpoints (parallel).
        Write-Host "Step 2: regenerating C_SFT clean rollouts..."
        modal run --detach modal_train.py sample_local -- `
          --model_path asingh15/qwen-sft-countdown-defaultproj `
          --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl `
          --output_json /vol/evaluation/eval_results/eval_c_sft_n500_fixed.json `
          --num_responses 16 --max_prompts 500 `
          --temperature 0.6 --top_p 0.95 --top_k 20 `
          --stop_strings '</answer>'

        Write-Host "Step 2: regenerating C_outcome clean rollouts..."
        modal run --detach modal_train.py sample_local -- `
          --model_path /vol/checkpoints/rloo_checkpoints/rloo_training/rloo_fixed_v2/latest_checkpoint/model `
          --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl `
          --output_json /vol/evaluation/eval_results/eval_c_outcome_n500_fixed.json `
          --num_responses 16 --max_prompts 500 `
          --temperature 0.6 --top_p 0.95 --top_k 20 `
          --stop_strings '</answer>'

        Write-Host "Both step-2 evals spawned. Watch progress on Modal dashboard or W&B."
    }

    "4b" {
        # Step 4b: probe-best-of-K RLOO (the recommended capstone run).
        Write-Host "Step 4b: launching probe_topk_M=4 full RLOO run from C_SFT (~5h, ~`$30)..."
        modal run --detach modal_train.py rloo -- `
          --model_name asingh15/qwen-sft-countdown-defaultproj `
          --ref_model_name asingh15/qwen-sft-countdown-defaultproj `
          --tokenizer_name asingh15/qwen-sft-countdown-defaultproj `
          --wandb_project rloo_probe_topk_0.5b `
          --wandb_name probe_topk4_csft_FULL `
          --save_dir /vol/checkpoints/rloo_probe_topk_checkpoints `
          --batch_size 128 `
          --group_size 8 `
          --gradient_accumulation_steps 128 `
          --num_training_steps 100 `
          --save_every_n_steps 10 `
          --warmup_ratio 0 `
          --lr_schedule constant `
          --probe_baseline `
          --probe_topk_M 4

        Write-Host "Spawned. Trainer should print '[probe_topk] ACTIVE: only top-4...' to confirm."
    }

    "5" {
        # Step 5: eval the finished probe_topk checkpoint.
        Write-Host "Step 5: eval probe_topk_M=4 checkpoint on n=500 with fixed sampler..."
        modal run --detach modal_train.py sample_local -- `
          --model_path /vol/checkpoints/rloo_probe_topk_checkpoints/rloo_probe_topk_0.5b/probe_topk4_csft_FULL/latest_checkpoint/model `
          --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl `
          --output_json /vol/evaluation/eval_results/eval_probe_topk4_FULL_n500.json `
          --num_responses 16 --max_prompts 500 `
          --temperature 0.6 --top_p 0.95 --top_k 20 `
          --stop_strings '</answer>'

        Write-Host "Eval spawned. Compare pass@1 vs vanilla C_outcome (~0.55) when done."
    }

    default {
        Write-Host "Unknown step: $Step"
        Write-Host "Valid steps: 2 (regen rollouts), 4b (probe_topk RLOO), 5 (eval probe_topk)"
        Write-Host "See RUNBOOK_CLEAN_REPLICATION.md for full pipeline."
        exit 1
    }
}
