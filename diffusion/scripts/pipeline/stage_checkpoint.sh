#!/usr/bin/env bash
# Stage 1c - assemble a sampling-ready checkpoint directory.
#
# WHY THIS EXISTS: sample_cond.py resolves `-r <file>` as logdir=dirname(file)
# and then loads "<logdir>/config.yaml". Training does NOT produce that layout -
# it writes checkpoints/epoch=NNNNNN.ckpt and configs/<timestamp>-project.yaml in
# separate trees. Without this step, sampling dies on a missing config.yaml.
#
# Usage:   bash scripts/pipeline/stage_checkpoint.sh LOGDIR [CKPT_NAME] [OUT_NAME]
#   e.g.   bash scripts/pipeline/stage_checkpoint.sh \
#            logs/semantickitti/2024-10-17T17-39-02_semantic_cond_c2_p4_wo_ls \
#            epoch=000052.ckpt config_w_weights
#          # last epoch instead of a chosen one:
#          bash scripts/pipeline/stage_checkpoint.sh <LOGDIR> last.ckpt config_w_weights_final
#
# Result:  <LOGDIR>/<OUT_NAME>/{config.yaml,model.ckpt}  <- pass this to sample_cond.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR="${1:?usage: stage_checkpoint.sh LOGDIR [CKPT_NAME] [OUT_NAME]}"
CKPT_NAME="${2:-last.ckpt}"
OUT_NAME="${3:-config_w_weights}"

SRC_CKPT="${LOGDIR}/checkpoints/${CKPT_NAME}"
[ -f "${SRC_CKPT}" ] || { echo "no such checkpoint: ${SRC_CKPT}"; ls "${LOGDIR}/checkpoints"; exit 1; }

SRC_CFG=$(ls -1 "${LOGDIR}"/configs/*-project.yaml 2>/dev/null | tail -1)
[ -n "${SRC_CFG}" ] || { echo "no *-project.yaml under ${LOGDIR}/configs"; exit 1; }

OUT="${LOGDIR}/${OUT_NAME}"
mkdir -p "${OUT}"
cp -v "${SRC_CKPT}" "${OUT}/model.ckpt"
cp -v "${SRC_CFG}"  "${OUT}/config.yaml"
echo ">>> staged: ${OUT}  (pass this dir, or its model.ckpt, to sample_cond.sh)"
