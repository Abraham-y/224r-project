# PowerShell version of scripts/eval_multiplicative_all_checkpoints.sh
# Spawns 10 Modal eval jobs (one per saved checkpoint: step_10 ... step_90 plus latest).
# Cost ~$22, ~2h.

$ErrorActionPreference = "Stop"

if (Test-Path ".env") {
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

$CKPT_BASE = "/vol/checkpoints/rloo_probe_mult_checkpoints/rloo_probe_mult_0.5b/probe_mult_csft_run1"
$EVAL_OUT_BASE = "/vol/evaluation/eval_results"

foreach ($STEP in 10,20,30,40,50,60,70,80,90) {
    Write-Host ">>> launching eval for step $STEP..."
    modal run --detach modal_train.py sample_local -- `
        --model_path "$CKPT_BASE/epoch_0_step_$STEP/model" `
        --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl `
        --output_json "$EVAL_OUT_BASE/eval_probe_mult_step${STEP}_n500.json" `
        --num_responses 16 --max_prompts 500 `
        --temperature 1.0 --top_p 1.0 --top_k -1 `
        --extra_stop_token_ids 151645
}

Write-Host ">>> launching final-checkpoint eval (latest_checkpoint)..."
modal run --detach modal_train.py sample_local -- `
    --model_path "$CKPT_BASE/latest_checkpoint/model" `
    --input_jsonl /root/default_proj/extension/data/countdown_eval_500.jsonl `
    --output_json "$EVAL_OUT_BASE/eval_probe_mult_final_n500.json" `
    --num_responses 16 --max_prompts 500 `
    --temperature 1.0 --top_p 1.0 --top_k -1 `
    --extra_stop_token_ids 151645

Write-Host ">>> all 10 evals spawned. monitor on Modal dashboard."
Write-Host ">>> when they finish, download to repo root and run: python scripts/compute_verifier_acc.py"
