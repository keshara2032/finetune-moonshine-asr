#!/bin/bash
#SBATCH --job-name=moonshine_eval
#SBATCH --output=./logs/eval-%x-%j.out
#SBATCH --error=./logs/eval-%x-%j.err
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
: "${PHASE:=1}"
: "${SPLIT:=test}"
: "${PHASE_DATASET:=/sfs/weka/scratch/cjh9fw/finetune-moonshine-asr}"
: "${FULL_DATASET:=/sfs/weka/scratch/cjh9fw/finetune-moonshine-asr}"
: "${FULL_WINDOW_SECONDS:=5}"
: "${FULL_WINDOW_STRIDE_SECONDS:=5}"
: "${SKIP_FULL_DATASET:=1}"
: "${MAX_SAMPLES:=}"

OUTPUT_BASE="/sfs/weka/scratch/cjh9fw/moonshine/results/${RUN_ID}/moonshine-en-accents"
PHASE_MODEL="${OUTPUT_BASE}-phase${PHASE}/final"
EVAL_CONFIG="${SLURM_TMPDIR:-/tmp}/moonshine_eval_${SLURM_JOB_ID:-manual}.yaml"

if [[ ! -d "$PHASE_MODEL" ]]; then
  echo "ERROR: Phase model directory does not exist: $PHASE_MODEL" >&2
  exit 1
fi

python - "$CONFIG" "$EVAL_CONFIG" "$OUTPUT_BASE" <<'PY'
import sys
from pathlib import Path

import yaml

source_config = Path(sys.argv[1])
eval_config = Path(sys.argv[2])
output_base = sys.argv[3]

with source_config.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.setdefault("training", {})["output_dir"] = output_base

with eval_config.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

EVAL_ARGS=(
  --config "$EVAL_CONFIG"
  --include-base
  --phases "$PHASE"
  --split "$SPLIT"
  --phase-dataset-path "$PHASE_DATASET"
  --phase-filter-phase "$PHASE"
  --full-dataset-path "$FULL_DATASET"
  --full-window-seconds "$FULL_WINDOW_SECONDS"
  --full-window-stride-seconds "$FULL_WINDOW_STRIDE_SECONDS"
)

if [[ "$SKIP_FULL_DATASET" == "1" ]]; then
  EVAL_ARGS+=(--skip-full-dataset)
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  EVAL_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "Evaluating phase ${PHASE} model:"
echo "  Run id:        $RUN_ID"
echo "  Model:         $PHASE_MODEL"
echo "  Dataset:       $PHASE_DATASET"
echo "  Split:         $SPLIT"
echo "  Skip full set: $SKIP_FULL_DATASET"
if [[ -n "$MAX_SAMPLES" ]]; then
  echo "  Max samples:   $MAX_SAMPLES"
else
  echo "  Max samples:   all phase-subset samples"
fi
echo "  Eval config:   $EVAL_CONFIG"

python -u scripts/test_model.py "${EVAL_ARGS[@]}"
