# Changelog

## 2026-08-15

- Added `scripts/train_kitti_slurm.sh`: Slurm batch script for SemanticKITTI training
  on 1 node x 2 GPUs, launching `main.py` via `torchrun` (conda env activation,
  per-job `MASTER_PORT`, mamba-ssm sanity check before launch).

- Removed `nuscenes-devkit==1.1.11` from both requirements files: it pins
  `matplotlib<3.6.0`, making the pin set unsatisfiable against `matplotlib==3.10.9`.
  The devkit is never imported by training/eval code (the nuScenes dataset reads a
  pre-generated JSON info file), so this unblocks install with no runtime impact.

Files touched: `scripts/train_kitti_slurm.sh`, `requirements.txt`,
`requirements-cu121-lock.txt`

Follow-ups:
- Paper setup is 4 GPUs x batch 6 (effective 24); the 2-GPU run halves the effective
  batch. Consider raising `training.batch_size` in the YAML or scaling `lr`.
