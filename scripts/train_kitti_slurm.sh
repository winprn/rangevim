#!/bin/bash
#SBATCH --job-name=rangevim-kitti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12          # >= 2 * num_workers per GPU, plus headroom
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --nodelist=gpu01
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults for this cluster; override any of them from the environment, e.g.
#   CONFIG=config/kitti/main/config_tinyvim_noaug.yaml sbatch scripts/train_kitti_slurm.sh
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-rangevim}"
DATA_ROOT="${DATA_ROOT:-/media02/nnthao10/dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences}"
PRETRAINED="${PRETRAINED:-/media02/nnthao10/pretrained/tinyvim_b.pth}"   # TinyViM ImageNet weights
CONFIG="${CONFIG:-config/kitti/main/config_tinyvim_noaug.yaml}"
SAVE_PATH="${SAVE_PATH:-./logs/rangevim_kitti_${SLURM_JOB_ID:-local}}"
# ---------------------------------------------------------------------------

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm_logs "$SAVE_PATH"

# Fail before the job burns allocation on a bad path.
[[ -d "$DATA_ROOT" ]]  || { echo "ERROR: DATA_ROOT not a directory: $DATA_ROOT" >&2; exit 1; }
[[ -f "$PRETRAINED" ]] || { echo "ERROR: PRETRAINED weights not found: $PRETRAINED" >&2; exit 1; }

# --- environment -----------------------------------------------------------
# No CUDA module is loaded on purpose. The spack modules top out at 11.8, while
# this env is built against CUDA 12.1 (driver 535.216.03 supports up to 12.2).
# The 12.1 toolkit lives inside the conda env; loading the 11.8 module here would
# put the wrong nvcc/libs ahead of it on PATH and LD_LIBRARY_PATH.
# Make `conda activate` usable from a non-interactive shell.
source "/home/nnthao10/.zshrc"
if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

# --- distributed rendezvous (single node, torchrun owns the env vars) ------
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
# Derive a per-job port so concurrent jobs on the same node don't collide.
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / (NGPUS > 0 ? NGPUS : 1) ))
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  gpus=${NGPUS}  env=${CONDA_ENV}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
# Fail fast on a mis-built env rather than 20 minutes into training.
python - <<'PY'
import torch, selective_scan_cuda  # noqa: F401  (import is the check)
assert torch.cuda.is_available(), 'CUDA not visible to torch'
assert torch.version.cuda.startswith('12.'), f'expected a CUDA 12.x build, got {torch.version.cuda}'
print(f'torch {torch.__version__} | cuda {torch.version.cuda} | mamba-ssm ok')
PY

# --- launch ----------------------------------------------------------------
srun --kill-on-bad-exit=1 torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py "${CONFIG}" \
    --data_root "${DATA_ROOT}" \
    --save_path "${SAVE_PATH}" \
    --pretrained_model "${PRETRAINED}" \
    --num_workers 4

echo "done. checkpoints + logs in ${SAVE_PATH}"
