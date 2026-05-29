#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

export WANDB__SERVICE_WAIT=300
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface}

base_model='Qwen/Qwen2.5-0.5B'

all_checkpoint_paths=(
    'PATH_TO_CHECKPOINT'
)
num_checkpoint_paths=${#all_checkpoint_paths[@]}

output_names=(
    'OUTPUT_NAME'
)
num_output_names=${#output_names[@]}

if [ $num_checkpoint_paths -ne $num_output_names ]; then
    echo "Number of checkpoint paths and output names do not match"
    exit 1
fi

for i in $(seq 0 $((num_checkpoint_paths - 1))); do
    checkpoint_path=${all_checkpoint_paths[$i]}
    output_name=${output_names[$i]}
    echo "Uploading model to $checkpoint_path to $output_name"
    command="HF_HUB_ENABLE_HF_TRANSFER=1 python sft_trainer/upload_sft.py --model_path $checkpoint_path --base_model $base_model --output_name $output_name"
    echo "Running command: $command"
    eval $command &
done
wait
echo "All models uploaded"