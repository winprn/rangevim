# Changelog

## 2026-08-16

- Regenerated `config/kitti/main/config_tinyvim_noaug.yaml` as a clone of
  `config_tinyvim_aug.yaml` with `range_aug: false` (range-image/pixel-level
  augmentation off; point-level geometric aug under `augmentation:` unchanged)
  and `data.use_trainval: true`. Gave it its own `id`/`save_path`/`run_name` so
  it no longer collides with the aug run's output directory.

- Added top-level `distributed: true` to that config only. `option.py:36` reads
  the key with a `False` default and `utils/tools/utils.py:54` returns early on
  a falsy value, so before this no run in the repo's history ever reached
  `init_process_group` — every rank trained independently on GPU 0 with no DDP
  wrap and no `DistributedSampler`.

Files touched: `config/kitti/main/config_tinyvim_noaug.yaml`

Follow-ups:
- The other configs (`config_tinyvim_aug.yaml`, `config_tinyvim_trainval.yaml`,
  `config/nusc/*`) still lack the key and remain effectively single-GPU.
- With DDP actually active, effective batch becomes `batch_size * nproc`; the
  paper's 4 x 6 = 24 now needs the GPU count to match.
- `use_trainval: true` folds seq 08 into the training set (`train.py:143`) while
  the val loader still evaluates on seq 08, so reported val mIoU is train-on-train
  and not a held-out number. Use it for test-server submissions only.

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
