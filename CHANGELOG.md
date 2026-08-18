# Changelog

## 2026-08-18

- Reduced `training.batch_size` (6 -> 4) in
  `config/kitti/main/config_tinyvim_noaug.yaml`. With `distributed: true` the
  value is per-GPU, and 6 full 64x2048 frames per rank exhausted a 40GB A100
  (OOM in the forward pass; AMP was already on via `use_fp16: true`). A trial run
  at 3 measured ~21GB, implying ~6GB/sample plus ~2-3GB fixed, so 4 (~27GB) fits
  with headroom.

- Scaled `training.lr` 3e-4 -> 2e-4 to match the reduced effective batch
  (2 GPUs x 4 = 8, was 12).

- Set `training.val_frequency` 1 -> 5 in the same config. With
  `use_trainval: true` seq 08 is folded into training while the val loader still
  evaluates on seq 08, so the metric is not held out and is not worth ~4070
  full-frame forward passes (plus kNN post-processing) every epoch. `main.py:331`
  saves `checkpoint.pth` independently of validation, and `--full` resumes from
  that file, so nothing downstream depends on the skipped epochs.

Files touched: `config/kitti/main/config_tinyvim_noaug.yaml`

Follow-ups:
- `best_miou_model.pth` now only updates on epochs where validation runs, and the
  mIoU behind it is train-on-train. Use `--val_only` against a held-out split for
  any number that goes in the thesis.
- The ~21GB measurement is training-only; the first validation pass (full-frame,
  `use_sliding_window: false`, plus kNN post-processing) had not run yet, so peak
  memory at the epoch boundary is still unverified.
- `train.py` has no gradient accumulation, so recovering the effective batch of
  12 at this memory footprint would need one (accumulate K micro-batches and
  wrap the non-final ones in `model.no_sync()`).
- The warmup/LR schedule (`warmup_epochs: 6`, 60 epochs) was not retuned for the
  new LR.

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
