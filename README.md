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

## Pretrained Weights

This repository does **not** include pretrained weights or trained Synapse checkpoints. Create `pre_trained_weights/` locally and place the downloaded files there.

TopoMamba-2D uses VMamba / VSSM ADE20K UperNet segmentation checkpoints as backbone initialization:

- Upstream project: https://github.com/MzeroMiko/VMamba
- Expected local files:
  - `pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth` for `TopoMamba_2D_t`
  - `pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth` for `TopoMamba_2D_s`
  - `pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth` for `TopoMamba_2D_b`

TopoMamba-3D uses a SegMamba-compatible 3D Mamba checkpoint as encoder initialization:

- Upstream project: https://github.com/ge-xing/SegMamba
- Expected local file:
  - `pre_trained_weights/tmp_model_ep799_0.8498.pt` for `TopoMamba_3D_t`

If you use a different checkpoint filename, update `pretrained_path` in `configs/config_setting_synapse.py` or pass `--load_pretrained false` to train from scratch.

## Synapse Data Layout

This repository does **not** ship Synapse data. Download the preprocessed Synapse files yourself and place them under `data/Synapse/` before training, testing, or running the included regression checks.

The public scripts follow the common TransUNet / Swin-Unet Synapse preprocessing convention. The processed Synapse/BTCV files used by many 2D baselines are linked from the Swin-Unet project:

- Swin-Unet project: https://github.com/HuCaoFighting/Swin-Unet
- Processed Synapse/BTCV data link from that README: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd

Please also follow the Synapse/BTCV dataset license and access requirements. If you prepare the dataset yourself, keep the same file names, list files, and array keys shown below.

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

## nnU-Net v2 Inspired 3D Protocol

The TopoMamba-3D Synapse pipeline is a lightweight, dataset-specific implementation inspired by nnU-Net v2 rather than a full nnU-Net fork. nnU-Net v2 provides a strong reference for self-configuring medical segmentation pipelines, including dataset analysis, preprocessing, patch planning, sliding-window inference, and post-processing.

For this public Synapse release, we keep the original Synapse file protocol used by TransUNet / Swin-Unet and add the nnU-Net-style parts needed for stable 3D training:

- case-level cache generation under `data/Synapse/topomamba3d_nnunetlite/`;
- train/test split manifests that preserve `lists/lists_Synapse/test_vol.txt`;
- shared 3D orientation, intensity clipping, normalization, and label handling for training and testing;
- foreground-aware 3D crop sampling for training patches;
- sliding-window inference with optional Gaussian blending;
- validation-gated connected-component post-processing selected only from held-out validation cases;
- a safety guard that keeps mirror TTA disabled for side-specific Synapse organs unless anatomy-aware class swapping is implemented.

The main model difference is that the segmentation network is `models/TopoMamba_3D.py`, not the nnU-Net U-Net. The preprocessing and inference code lives in `tools/synapse3d_preprocess.py` and `tools/synapse3d_pipeline.py`.

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

## Acknowledgements

TopoMamba builds on ideas and public resources from several excellent projects. We thank the authors and maintainers of VMamba for the VSSM backbone and ADE20K checkpoints, SegMamba for 3D Mamba medical segmentation references, nnU-Net / nnU-Net v2 for the self-configuring medical segmentation pipeline design, and the TransUNet / Swin-Unet communities for the widely used Synapse preprocessing protocol.

