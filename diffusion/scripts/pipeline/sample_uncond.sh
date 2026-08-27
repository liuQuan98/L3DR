#!/usr/bin/env bash
# Stage 1d' - unconditional sampling (used for the diffusion-agnostic inference
# experiments, where an RRN trained on semantic-conditioned data rectifies output
# from an unconditional model).
#
# Usage:   bash scripts/pipeline/sample_uncond.sh CKPT [N_SAMPLES] [METRIC_SET]
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?usage: sample_uncond.sh CKPT [N_SAMPLES] [METRIC_SET]}"
N="${2:-2000}"
DSET="${3:-kitti}"

echo ">>> unconditional sampling: ${CKPT}  n=${N}"
python scripts/sample.py -d "${DSET}" -r "${CKPT}" -n "${N}" --eval
echo ">>> done."

# previously run:
#   python scripts/sample.py -d kitti -r models/lidm/kitti/uncond_wo_logscale/model.ckpt -n 19130
