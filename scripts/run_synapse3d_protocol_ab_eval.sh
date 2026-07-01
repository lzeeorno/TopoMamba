#!/usr/bin/env bash
set -euo pipefail

# TopoMamba-3D Synapse Protocol A/B evaluation.
#
# This script runs the validated public evaluation path:
#   1. inference on held-out validation cases,
#   2. validation-gated postprocess selection,
#   3. final inference on Synapse test_vol cases.
#
# Mirror TTA is intentionally disabled. Synapse has side-specific organs
# (Left_Kidney / Right_Kidney), and all-axis mirror TTA is unsafe unless
# anatomy-aware class-channel swaps are implemented and validated.

NETWORK="${NETWORK:-TopoMamba_3D_t}"
WEIGHTS="${WEIGHTS:-results/TopoMamba_3D_t_synapse_protocolA_B_retrain/checkpoints/best.pth}"
CACHE_ROOT="${CACHE_ROOT:-${SYNAPSE3D_CACHE_ROOT:-data/Synapse/topomamba3d_nnunetlite}}"
DEVICE="${DEVICE:-${SYNAPSE3D_DEVICE:-cuda}}"
VAL_CASES="${VAL_CASES:-case0031,case0007,case0009,case0005,case0026,case0039}"
RUN_NAME="${RUN_NAME:-TopoMamba_3D_t_synapse_protocolA_B}"
GAUSSIAN_BLENDING="${SYNAPSE3D_GAUSSIAN_BLENDING:-0}"
SAVE_VISUALIZATIONS="${SYNAPSE3D_SAVE_VISUALIZATIONS:-0}"

VAL_OUT_DIR="${VAL_OUT_DIR:-test_results/${RUN_NAME}_val_noaug}"
TEST_OUT_DIR="${TEST_OUT_DIR:-test_results/${RUN_NAME}_test_noaug}"
POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-results/TopoMamba_3D_t_synapse_protocolA_B_retrain/postprocess_config_noaug.json}"

echo "[TopoMamba3D] network: ${NETWORK}"
echo "[TopoMamba3D] weights: ${WEIGHTS}"
echo "[TopoMamba3D] cache: ${CACHE_ROOT}"
echo "[TopoMamba3D] validation cases: ${VAL_CASES}"
echo "[TopoMamba3D] gaussian blending: ${GAUSSIAN_BLENDING}"
echo "[TopoMamba3D] mirror TTA: disabled"

IFS=',' read -r -a VAL_CASE_ARRAY <<< "${VAL_CASES}"

SYNAPSE3D_CACHE_ROOT="${CACHE_ROOT}" \
SYNAPSE3D_DEVICE="${DEVICE}" \
SYNAPSE3D_EVAL_SPLIT=train \
SYNAPSE3D_CASES="${VAL_CASES}" \
SYNAPSE3D_GAUSSIAN_BLENDING="${GAUSSIAN_BLENDING}" \
SYNAPSE3D_MIRROR_TTA=0 \
SYNAPSE3D_OUT_DIR="${VAL_OUT_DIR}" \
SYNAPSE3D_SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS}" \
python -u test_synapse.py \
  --network "${NETWORK}" \
  --weights "${WEIGHTS}"

python tools/select_synapse3d_postprocess.py \
  --pred-dir "${VAL_OUT_DIR}/predictions" \
  --label-cache-dir "${CACHE_ROOT}/train_cases" \
  --out "${POSTPROCESS_CONFIG}" \
  --cases "${VAL_CASE_ARRAY[@]}"

SYNAPSE3D_CACHE_ROOT="${CACHE_ROOT}" \
SYNAPSE3D_DEVICE="${DEVICE}" \
SYNAPSE3D_EVAL_SPLIT=test \
SYNAPSE3D_GAUSSIAN_BLENDING="${GAUSSIAN_BLENDING}" \
SYNAPSE3D_MIRROR_TTA=0 \
SYNAPSE3D_POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG}" \
SYNAPSE3D_OUT_DIR="${TEST_OUT_DIR}" \
SYNAPSE3D_SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS}" \
python -u test_synapse.py \
  --network "${NETWORK}" \
  --weights "${WEIGHTS}"

echo "[TopoMamba3D] done. Results: ${TEST_OUT_DIR}"
