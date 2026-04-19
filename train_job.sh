#!/bin/bash
#SBATCH --job-name=qwen3_tts_synth
#SBATCH --output=./logs/slurm-%x-%A_%a.out
#SBATCH --error=./logs/slurm-%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

python -u train.py --config configs/english_accent_local_no_curriculum.yaml

echo "Training completed on host: ${HOSTNAME}"