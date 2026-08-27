#!/usr/bin/env bash
# Stage 1b - train the latent diffusion model.
#
# Usage:   bash scripts/pipeline/train_diffusion.sh [CONFIG]
#          GPUS=0,1,2,3 bash scripts/pipeline/train_diffusion.sh
#
# Prereq:  the config's first_stage_config.params.ckpt_path must point at a trained
#          autoencoder checkpoint (see train_autoencoder.sh).
# Output:  logs/<dataset>/<timestamp>_<config>/
# Next:    sample_cond.sh - it stages the checkpoint and samples both splits.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/lidar_diffusion/semantickitti/semantic_cond_c2_p4_wo_ls.yaml}"
GPUS="${GPUS:-0,1,2,3}"

echo ">>> training diffusion: ${CONFIG}  (gpus ${GPUS})"
python main.py -b "${CONFIG}" -t --gpus "${GPUS}"

echo ">>> training done: ${CONFIG}"
echo
echo ">>> next, sample both splits (this stages the checkpoint for you):"
echo "      bash scripts/pipeline/sample_cond.sh logs/$(basename "$(dirname "${CONFIG}")")/<timestamp>_$(basename "${CONFIG}" .yaml)"

# previously run:
#   python main.py -b configs/lidar_diffusion/semantickitti/semantic_cond_c2_p4_wo_ls.yaml               -t --gpus 0,1,2,3
#   python main.py -b configs/lidar_diffusion/semantickitti/semantic_uncond_c2_p4_wo_ls_w_seg_input.yaml -t --gpus 0,1,2,3
#   python main.py -b configs/lidar_diffusion/semantickitti/clip_cond_c2_p4_wo_ls.yaml                   -t --gpus 1,2,3
#   python main.py -b configs/lidar_diffusion/waymo/semantic_cond_c2_p4_wo_ls.yaml                       -t --gpus 1,2,3
#   python main.py -b configs/lidar_diffusion/nuscenes/semantic_cond_c2_p4_wo_ls.yaml                    -t --gpus 1,2,3
#
# resume:
#   python main.py -b <config> -t --gpus 0, -r <logdir-or-ckpt>
