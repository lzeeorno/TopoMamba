# TopoMamba

Official implementation for **TopoMamba: Topology-Aware Scanning and Fusion for Medical Image Segmentation**.


## Citation

If you use this code, please cite the TopoMamba paper. 

```bash
@article{zheng2026topomamba,
  title={TopoMamba: Topology-Aware Scanning and Fusion for Segmenting Heterogeneous Medical Visual Media},
  author={Zheng, Fuchen and Xu, Chengpei and Ma, Long and Li, Weixuan and Zhou, Junhua and Chen, Xuhang and Liu, Weihuang and Li, Haolun and Li, Quanjun and Zhang, Zhenxi and others},
  journal={arXiv preprint arXiv:2604.25545},
  year={2026}
}
```



This public release focuses on the Synapse multi-organ CT segmentation experiments for:

- `models/TopoMamba_2D.py`
- `models/TopoMamba_3D.py`

The repository intentionally excludes datasets, checkpoints, experiment logs, ablation-only code, result folders, and paper build artifacts.

## Highlights

- **TopoMamba-2D**: slice-level medical image segmentation with topology-aware scan ordering, ScanCache, and HSIC-based branch fusion.
- **TopoMamba-3D**: volumetric Synapse segmentation with 3D scan order, case-level cache construction, foreground crop training, sliding-window inference, and validation-gated post-processing.
- **Synapse-compatible protocol**: the 3D path is built from the standard Synapse `train_npz` and `test_vol_h5` files while preserving the official `test_vol.txt` case order.

## Environment

The code was tested with:

- Python 3.8
- PyTorch 2.0.1 + CUDA 11.7
- One NVIDIA RTX 4090 for the reported Synapse3D run

Create and activate the environment:

```bash
conda create -n cmamba python=3.8.20 -y
conda activate cmamba
pip install -r requirements.txt
```

If your PyTorch/CUDA stack is different, install the matching PyTorch build first, then install the remaining requirements.

## Synapse Data Layout

This repository does **not** ship Synapse data. Download the preprocessed Synapse files yourself and place them under `data/Synapse/` before training, testing, or running the included regression checks.

Prepare Synapse in the commonly used Swin-Unet / nnFormer style:

```text
data/Synapse/
├── train_npz/
│   ├── case0001_slice000.npz
│   ├── case0001_slice001.npz
│   ├── case0002_slice000.npz
│   └── ...
├── test_vol_h5/
│   ├── case0001.npy.h5
│   ├── case0002.npy.h5
│   └── ...
└── lists/lists_Synapse/
    ├── train.txt       # one training slice id per line, e.g. case0031_slice000
    └── test_vol.txt    # one testing volume id per line, e.g. case0008
```

Each `train_npz/*.npz` file must contain `image` and `label` arrays for a 2D slice. Each `test_vol_h5/*.npy.h5` file must contain `image` and `label` datasets for a 3D volume with shape `(D, H, W)`.

The 3D pipeline derives a case-level cache on first use:

```text
data/Synapse/topomamba3d_nnunetlite/
├── manifests/
├── train_cases/
└── test_cases/
```

This derived cache is generated automatically and is ignored by git.

## Train TopoMamba-2D on Synapse

```bash
conda activate cmamba

python train_synapse.py \
  --network TopoMamba_2D_t \
  --work_dir results/TopoMamba_2D_t_synapse
```

Available 2D variants:

```bash
python train_synapse.py --network TopoMamba_2D_t
python train_synapse.py --network TopoMamba_2D_s
python train_synapse.py --network TopoMamba_2D_b
```

Test a 2D checkpoint:

```bash
python test_synapse.py \
  --network TopoMamba_2D_t \
  --weights results/TopoMamba_2D_t_synapse/checkpoints/best.pth
```

## Train TopoMamba-3D on Synapse

The 3D entrypoint is routed through the same `train_synapse.py` file when the network is `TopoMamba_3D_t`.

```bash
VAL_CASES_COMMA="case0031,case0007,case0009,case0005,case0026,case0039"

SYNAPSE3D_CACHE_ROOT=data/Synapse/topomamba3d_nnunetlite \
SYNAPSE3D_DEVICE=cuda \
SYNAPSE3D_REBUILD_CACHE=1 \
SYNAPSE3D_TRAIN_EXCLUDE_CASES="$VAL_CASES_COMMA" \
PYTHONUNBUFFERED=1 \
python -u train_synapse.py \
  --network TopoMamba_3D_t \
  --work_dir results/TopoMamba_3D_t_synapse_protocolA_B_retrain \
  --epochs 1000 \
  --steps_per_epoch 250 \
  --resume false \
  --save_interval 100 \
  --print_interval 20
```

The held-out validation cases are excluded from training and later used only for validation-gated post-processing.

## Evaluate TopoMamba-3D on Synapse

Use the packaged evaluation script:

```bash
bash scripts/run_synapse3d_protocol_ab_eval.sh
```

By default, it runs:

1. validation inference on held-out training cases,
2. post-processing selection using validation labels,
3. final inference on `test_vol.txt` test cases.

Override common paths with environment variables:

```bash
WEIGHTS=results/TopoMamba_3D_t_synapse_protocolA_B_retrain/checkpoints/best.pth \
CACHE_ROOT=data/Synapse/topomamba3d_nnunetlite \
RUN_NAME=TopoMamba_3D_t_synapse_protocolA_B \
bash scripts/run_synapse3d_protocol_ab_eval.sh
```

The validated no-augmentation test path produced:

```text
Average Dice: 0.9344 ± 0.0346
Mean HD95:    4.8255 ± 7.5819
Output:       test_results/TopoMamba_3D_t_synapse_protocolA_B_test_noaug
```

## Important Inference Note

Keep mirror TTA disabled for Synapse3D main results:

```text
SYNAPSE3D_MIRROR_TTA=0
```

Synapse includes side-specific organs such as `Left_Kidney` and `Right_Kidney`. All-axis mirror TTA is not label-equivariant unless left/right class channels are swapped correctly. The evaluator guards against unsafe all-axis mirror TTA by default.

Gaussian blending can be evaluated separately:

```bash
SYNAPSE3D_GAUSSIAN_BLENDING=1 \
bash scripts/run_synapse3d_protocol_ab_eval.sh
```

For the validated public command above, no mirror TTA is used.

## Outputs

Training outputs:

```text
results/<run_name>/
├── checkpoints/
├── train_record.csv
└── train3d_summary.json
```

Evaluation outputs:

```text
test_results/<run_name>/
├── predictions/
├── test_results_detailed.json
├── test_results_summary.csv
└── organ_metrics.md
```


