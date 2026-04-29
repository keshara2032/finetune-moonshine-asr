#!/bin/bash
#SBATCH --job-name=moonshine_finetune
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --account="uva-dsa"

# Uncomment and adapt these lines for your environment.
module purge
module load miniforge
module load gcc/11.4.0
module load cuda/13.0.2
source /home/cjh9fw/.bashrc
echo "[setup_env] Host: $HOSTNAME"

conda deactivate || true
conda activate moonshine

echo "Host: ${HOSTNAME}"

# Author-aligned default: no-curriculum training from the config.
# Resume is opt-in, e.g.:
#   sbatch --export=ALL,RESUME=1,RUN_ID=123456 train_job.sh
RESUME="${RESUME:-0}"
RESUME_RUN_ID="${RESUME_RUN_ID:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PHASE="${PHASE:-}"

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]] && [[ -n "${RESUME_RUN_ID}" ]]; then
  RUN_ID="${RUN_ID:-${RESUME_RUN_ID}}"
else
  RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-local}}"
fi

OUTPUT_DIR="results/${RUN_ID}/moonshine-en-accents"
TRAIN_ARGS=(
  --config configs/english_accent_local_no_curriculum.yaml
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${PHASE}" ]]; then
  TRAIN_ARGS+=(--phase "${PHASE}")
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]]; then
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_CHECKPOINT}")
  else
    TRAIN_ARGS+=(--resume)
  fi
fi

echo "Mode: no-curriculum config (PHASE is ignored unless curriculum.enabled=true)"
echo "Phase override: ${PHASE:-none}"
echo "Resume: ${RESUME}"
echo "Run id: ${RUN_ID}"
echo "Output dir: ${OUTPUT_DIR}"

python -u train.py "${TRAIN_ARGS[@]}"

echo "Training completed on host: ${HOSTNAME}"
