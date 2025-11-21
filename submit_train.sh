#!/bin/bash
#SBATCH --job-name=esm_train
#SBATCH --partition=pi_krishnaswamy
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -euo pipefail
mkdir -p ../sbatch_logs

# Modules / env
module --force purge
module load StdEnv
module load miniconda

# Activate conda environment
set +u
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate imageflownet_gpu
set -u

# Repo root on this cluster
REPO_ROOT="/gpfs/gibbs/pi/krishnaswamy_smita/sa2556/esm_prototype"
cd "$REPO_ROOT"

# Diagnostics
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__, "built cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# Run training
python train.py 
