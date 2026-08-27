#!/usr/bin/env bash
# Stage 3 - score the rectified point clouds coming back from ../rectification/.
#
# Reads the .npz files the RRN tester wrote, applies
#   P_rectified = P_gen + P_gen/||P_gen|| * predicted_offset
# and reports FRID / FSVD / FPVD / JSD / MMD.
#
# Usage:   bash scripts/pipeline/eval_refined_coords.sh RESULT_DIR [DATASET] [MODE]
#   e.g.   bash scripts/pipeline/eval_refined_coords.sh \
#            ../rectification/exp/generated_to_real_refinement/my-run/result_val kitti both
#
# MODE:  refined   score the rectified cloud
#        baseline  score the unrectified cloud from the same files
#        both      run baseline then refined  (default; this is how a table row pair is made)
#
# Env:   ORIGINAL_GT=1   take GT from the raw dataset rather than from the .npz.
#                        Needs the dataset/ symlinks and
#                        models/baseline/other_baselines/<dataset>.yaml.
#                        Compares against UNRESAMPLED GT, so it is the more
#                        trustworthy of the two comparisons.
#        NUM_FILES=N     cap how many frames are scored.
#        METRICS=a,b     which metrics to compute (default frid,fsvd,fpvd,jsd,mmd).
#                        frid/fsvd/fpvd need the pretrained_weights backbones and are
#                        slow; mmd is O(n^2) pairwise chamfer. `METRICS=jsd` is quick.
set -euo pipefail
cd "$(dirname "$0")/../.."

RESULT_DIR="${1:?usage: eval_refined_coords.sh RESULT_DIR [DATASET] [MODE]}"
DSET="${2:-kitti}"
MODE="${3:-both}"

[ -d "${RESULT_DIR}" ] || { echo "no such result dir: ${RESULT_DIR}"; exit 1; }
EXTRA=()
[ "${ORIGINAL_GT:-0}" = "1" ] && EXTRA+=(--eval_with_original_GT)
[ -n "${NUM_FILES:-}" ] && EXTRA+=(--num_files "${NUM_FILES}")
[ -n "${METRICS:-}" ] && EXTRA+=(--metrics "${METRICS}")

run () {  # $1 = "" | --eval_refined_coords
  echo ">>> scoring ${RESULT_DIR}  dataset=${DSET}  ${1:-（baseline）}"
  python scripts/eval_refined_coords.py -d "${DSET}" --eval --file "${RESULT_DIR}" \
         ${1:+$1} "${EXTRA[@]+"${EXTRA[@]}"}"
}

case "${MODE}" in
  baseline) run "" ;;
  refined)  run --eval_refined_coords ;;
  both)     run ""; echo; run --eval_refined_coords ;;
  *) echo "MODE must be baseline|refined|both, got '${MODE}'"; exit 1 ;;
esac

# previously run:
#   python scripts/eval_refined_coords.py -d kitti    --eval --file <RESULT_DIR> --eval_refined_coords
#   python scripts/eval_refined_coords.py -d nuscenes --eval --file <RESULT_DIR> --eval_with_original_GT
#   python scripts/eval_refined_coords.py -d waymo    --eval --file <RESULT_DIR> --eval_refined_coords --eval_with_original_GT
#   python scripts/eval_refined_coords.py -d kitti360 --eval --file <RESULT_DIR> --eval_with_original_GT --num_files 1040
#   python scripts/eval_refined_coords.py -d kitti    --eval --file <RESULT_DIR> --eval_with_original_GT --num_files 4071  # R2DM
#   # per-point rather than distributional metrics:
#   python scripts/eval_refined_coords_pointwise_metrics.py -d kitti --eval --file <RESULT_DIR> [--eval_refined_coords]
#   # paper Figure 2 (RV image-gradient distribution):
#   python scripts/eval_img_grad_distribution.py -d kitti --file <RESULT_DIR> --eval_refined_coords --eval \
#          --filename <out>.npz --num_samples 1000 --num_grad_per_sample 100
