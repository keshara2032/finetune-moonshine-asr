#!/bin/bash
#SBATCH --job-name=moonshine_ckpt_eval
#SBATCH --output=./logs/debug-%x-%j.out
#SBATCH --error=./logs/debug-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --account="uva-dsa"

module purge
module load miniforge
module load gcc/11.4.0
module load cuda/13.0.2
source /home/cjh9fw/.bashrc

conda deactivate || true
conda activate moonshine

: "${CONFIG:=configs/english_accent_local_no_curriculum.yaml}"
: "${RUN_ID:=12216527}"
: "${PHASE:=2}"
: "${SPLIT:=test}"
: "${MAX_SAMPLES:=500}"
: "${CHECKPOINTS:=checkpoint-200 checkpoint-1000 checkpoint-4000 final}"
: "${DATASET_PATH:=}"
: "${DEVICE:=cuda}"
: "${FP16:=0}"
: "${MAX_NEW_TOKENS:=}"

ARGS=(
  --config "$CONFIG"
  --run-id "$RUN_ID"
  --phase "$PHASE"
  --split "$SPLIT"
  --max-samples "$MAX_SAMPLES"
  --device "$DEVICE"
  --checkpoints $CHECKPOINTS
)

if [[ -n "$DATASET_PATH" ]]; then
  ARGS+=(--dataset-path "$DATASET_PATH")
fi

if [[ "$FP16" == "1" || "$FP16" == "true" || "$FP16" == "True" ]]; then
  ARGS+=(--fp16)
fi

if [[ -n "$MAX_NEW_TOKENS" ]]; then
  ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

echo "Debug checkpoint evaluation"
echo "  Config:      $CONFIG"
echo "  Run id:      $RUN_ID"
echo "  Phase:       $PHASE"
echo "  Split:       $SPLIT"
echo "  Max samples: $MAX_SAMPLES"
echo "  Checkpoints: $CHECKPOINTS"
echo "  Device:      $DEVICE"

python -u scripts/debug_checkpoint_eval.py "${ARGS[@]}"

