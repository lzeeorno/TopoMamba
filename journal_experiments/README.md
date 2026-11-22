# TopoMamba journal pilot (research-only; not paper results)

This branch intentionally leaves published models/configs unchanged. It starts a reproducible **falsification-first** study of scan ordering and HSIC. The experiment is NOT a reproducing result of ACMMM tables, nor a claimed TPAMI method.

## First audit: verify original run artifacts before new architectures

1. Paper Eq. (gate) assigns larger HSIC weight to TopoA, while `models/TopoMamba_2D.py::hsic_gate_C` currently assigns it to `traditional_out` (Cross). These are different operations. Determine which version produced Table 4/5.
2. The 2D code stores `self.hsic_alpha=float(...)`; the paper calls the scale learnable. Confirm actual checkpoint and training script.
3. The current 3D code labels `MiniBiMambaV3` a checkpoint-compatible approximation, not a full recurrent SSM. Recover the exact original 3D implementation/checkpoint before any new 3D comparison.
4. The optional `OptimizedHSICGate` in 2D source is NOT selected by the default `fusion_method='hsic'`; do not label its exploratory claims as validated.
5. Fix identical normalization, fusion scale, parameter counts, seeds, data split, initialization, training schedule across controls.

## Run the minimal experiments

```bash
python journal_experiments/scan_hsic_pilot.py
# Requires CUDA, compatible PyTorch, official mamba_ssm:
python journal_experiments/backend_probe.py --backend mamba2
python journal_experiments/backend_probe.py --backend mamba3
```

The first script reports synthetic geometry and a gate check; it does NOT train segmentation. The second script performs **random-initialized** SISO Mamba2/Mamba3 shape-and-gradient smoke tests (not pretrained checkpoints and not full segmentation). CUDA absence is reported as SKIPPED.

## Experimental gates

- G0: reproduce the *actual* conference baseline and identify paper/code drift. Do not compare against manuscript numbers until resolved.
- G1: fixed 0.5/0.5, fixed 0.6/0.4, learned global scalar, MLP, cosine/CKA, true HSIC, shuffled HSIC and local HSIC. Same trained setup; report alpha/HSIC/weight distributions and 3+ seed CIs.
- G2: factorial scan × state space operator: [Cross, TopoA, dual] × [VMamba, Mamba2, Mamba3-SISO]; identical data/training, report Dice/clDice/topology/throughput.
- G3: only if scan-path distortion predicts improved connectivity, test learned content-adaptive scan. Validate bijective permutation, gather/scatter, differentiability or estimator bias, and overhead.
- G4: only after G1–G3, optional DINOv3 ViT-S teacher distillation; no teacher at inference and a matched teacher-only baseline.
- Use DRIVE/CHASE_DB1 and Massachusetts Roads/DeepGlobe road before broad natural-scene segmentation; preserve Synapse/ISIC/CVC as historical anchors.

## Sources/weights (do not confuse language LM with image pretraining)

- Conference repo: https://github.com/lzeeorno/TopoMamba
- Official Mamba2/3: https://github.com/state-spaces/mamba (modules/mamba2.py, modules/mamba3.py).
- Language LM checkpoints: https://huggingface.co/state-spaces/models ; **not compatible vision pretraining** by default.
- DINOv3: https://github.com/facebookresearch/dinov3 ; gated weights at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and satellite ViT-L at https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m (access approval needed).
- Official DSCNet: https://github.com/YaoleiQi/DSCNet.

## Non-claims
Local normalized kernel dependence measures similarity, not task reliability. An HSIC label does NOT prove causal direction, topological invariance or an estimator error bound. Mamba-3's language/state-tracking gains do not imply superior 2D/3D segmentation.
