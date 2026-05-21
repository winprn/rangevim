# Configuration Files

All experiment configurations are organized under this directory.

## Structure

```
config/
├── README.md                     ← This file
├── kitti/
│   ├── main/                    Main RangeViM experiments (TinyViM + FPN)
│   ├── rangevit/                Original RangeViT configs (ViT-S + ConvStem + UpConv + KPConv)
│   ├── ablation/                Ablation studies
│   └── reproduce/               Paper reproduction configs
└── nusc/
    └── config_nusc_tinyvim_full.yaml  nuScenes TinyViM-Base full-frame experiment
```

---

## SemanticKITTI — `kitti/`

### RangeViT baseline (`rangevit/`)

Original RangeViT (ViT-S + ConvStem + UpConv decoder + KPConv 3D refiner), based on [Ando et al., CVPR 2023](https://arxiv.org/abs/2304.07693).

| File | Backbone | Decoder | Epochs | Batch | Resolution | Notes |
|------|----------|---------|--------|-------|------------|-------|
| `config_original.yaml` | ViT-Small | UpConv | 60 | 4 | 64x384 sliding-window | Original RangeViT baseline |
| `config_trainval.yaml` | ViT-Small | UpConv | 60 | 4 | 64x384 sliding-window | For test submission (train on 00-10) |

### RangeViM main experiments (`main/`)

| File | Backbone | Decoder | Epochs | Batch | Resolution | Augmentation | Notes |
|------|----------|---------|--------|-------|------------|-------------|-------|
| `config_tinyvim_aug.yaml` | TinyViM-Base | FPN | 60 | 6 | 64x2048 full-frame | Yes | Main paper result (65.96% val mIoU) |
| `config_tinyvim_noaug.yaml` | TinyViM-Base | FPN | 60 | 6 | 64x2048 full-frame | No | No-augmentation ablation |
| `config_tinyvim_trainval.yaml` | TinyViM-Base | FPN | 60 | 1 | 64x2048 full-frame | Yes | For test submission (train on 00-10) |

### Ablation studies (`ablation/`)

#### Backbone variants (`backbone/`)

| File | Backbone | Decoder | Resolution | Notes |
|------|----------|---------|------------|-------|
| `config_tinyvim_small.yaml` | TinyViM-Small | FPN | 64x2048 full-frame | 8.49M params, 65.00% val mIoU |
| `config_tinyvim_large.yaml` | TinyViM-Large | FPN | 64x2048 full-frame | 24.28M params, 65.07% val mIoU |

#### Decoder variants (`decoder/`)

| File | Decoder | Notes |
|------|---------|-------|
| `config_fpn_residual_cross_attn.yaml` | FPN-Residual-CrossAttn | Both residual + cross-attention |
| `config_fuse_aux.yaml` | Fuse-Aux | Simple concatenation + refinement baseline |

#### Sliding window studies (`window/`)

| File | Resolution | Mode | Notes |
|------|------------|------|-------|
| `config_64x1024.yaml` | 64x1024 | Sliding window | Overlapping (stride 512) and non-overlapping |
| `config_64x512.yaml` | 64x512 | Sliding window | 64x512 crops with overlap |

#### Robustness sensitivity (`robustness/`)

| File | Corruption | Severity |
|------|-----------|----------|
| `config_robust_point_dropout.yaml` | Projected-pixel dropout (p=0.10) | Validation-time only |
| `config_robust_beam_dropout.yaml` | Beam dropout (p=0.10) | Validation-time only |
| `config_robust_range_noise.yaml` | Normalized range-coordinate noise (sigma=0.03) | Validation-time only |

### Paper reproduction (`reproduce/`)

| File | Purpose | Expected Result |
|------|---------|----------------|
| `config_knn7.yaml` | Main config with KNN search window = 7 (paper value) | ~65.96% val mIoU |

---

## nuScenes — `nusc/`

| File | Backbone | Decoder | Epochs | Batch | Resolution | Notes |
|------|----------|---------|--------|-------|------------|-------|
| `config_nusc_tinyvim_full.yaml` | TinyViM-Base | FPN | 120 | 6 | 32x2048 full-frame | Main paper result (76.88% val mIoU) |

---

## Checkpoints

Pre-trained model checkpoints are available on Google Drive:

- [HCMUS-THESIS-RANGEVIM](https://drive.google.com/drive/folders/17zkW0KQPqzc87A2Ws30D25QHIVFqzpW5?usp=drive_link) — contains `nuscene/` and `SemanticKITTI/` subfolders

---

## Usage

### Training

```bash
# SemanticKITTI main experiment
python main.py config/kitti/main/config_tinyvim_aug.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/rangevim_kitti

# nuScenes
python main.py config/nusc/config_nusc_tinyvim_full.yaml \
    --data_root <NUSCENES_ROOT> \
    --save_path ./logs/rangevim_nusc
```

### Evaluation

```bash
python main.py config/kitti/main/config_tinyvim_aug.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/eval \
    --checkpoint <CHECKPOINT_PATH> \
    --val_only
```

### Profiling

```bash
python tools/profile_metrics.py config/kitti/main/config_tinyvim_aug.yaml \
    --device cuda --amp --validation_style \
    --batch_size 1 --warmup 20 --iters 50
```

---

## Key hyperparameters (matching the paper)

| Parameter | SemanticKITTI | nuScenes |
|-----------|---------------|----------|
| Epochs | 60 | 120 |
| Batch size (per GPU) | 6 | 6 |
| Peak learning rate | 3e-4 | 3e-4 |
| Warmup epochs | 6 | 10 |
| KNN search window | **7** | **7** |
| KNN k neighbors | 5 | 5 |
| KNN Gaussian sigma | 1.0 | 1.0 |
| KNN cutoff | 1.0 | 1.0 |
