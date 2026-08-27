#!/usr/bin/env bash
# Stage 1a - train the VQ-VAE range-image compressor (the LiDM "first stage").
#
# Usage:   bash scripts/pipeline/train_autoencoder.sh [CONFIG]
#          GPUS=0,1,2,3 bash scripts/pipeline/train_autoencoder.sh
#
# Output:  logs/<dataset>/<timestamp>_<config-name>/{checkpoints,configs,images}/
# Next:    stage_checkpoint.sh, then point the diffusion config's
#          first_stage_config.params.ckpt_path at the staged model.ckpt.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/autoencoder/semantickitti/autoencoder_c2_p4_wo_ls.yaml}"
GPUS="${GPUS:-0,1,2,3}"

echo ">>> training autoencoder: ${CONFIG}  (gpus ${GPUS})"
python main.py -b "${CONFIG}" -t --gpus "${GPUS}"
echo ">>> done: ${CONFIG}"

# previously run:
#   python main.py -b configs/autoencoder/semantickitti/autoencoder_c2_p4_wo_ls.yaml       -t --gpus 0,1,2,3
#   python main.py -b configs/autoencoder/semantickitti/autoencoder_c2_p4_wo_ls_w_seg.yaml -t --gpus 0,1,2,3
#   python main.py -b configs/autoencoder/waymo/autoencoder_c2_p4_wo_ls.yaml               -t --gpus 1,2,3
#   python main.py -b configs/autoencoder/nuscenes/autoencoder_c2_p4_wo_ls.yaml            -t --gpus 1,2,3
