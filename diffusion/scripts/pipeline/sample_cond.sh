#!/usr/bin/env bash
# Stage 1d - conditional sampling. Produces the generated/GT pairs that the
# rectification half trains on.
#
# Usage:   bash scripts/pipeline/sample_cond.sh LOGDIR_OR_CKPT [SPLIT] [METRIC_SET]
#          SPLIT is train | validation | both   (default: both)
#   e.g.   bash scripts/pipeline/sample_cond.sh <LOGDIR>              # both splits
#          bash scripts/pipeline/sample_cond.sh <LOGDIR> train        # just one
#
# You may also pass a RAW TRAINING LOGDIR - it is staged automatically (see
# stage_checkpoint.sh). Override which checkpoint / dir name is used with:
#   STAGE_CKPT=epoch=000052.ckpt STAGE_NAME=config_w_weights_final \
#     bash scripts/pipeline/sample_cond.sh <LOGDIR> train
#
# NOTE ON --split: it indexes config['data']['params'][split], so the only valid
#   values are the config's own keys - `train` and `validation`. `val` fails.
#   The default is `validation`, so the TRAIN-split run is easy to forget - and
#   that is the one the RRN trains on. Run both.
#
# NOTE ON -d: only `kitti` and `nuscenes` are valid - anything else KeyErrors in
#   DATASET2METRICS/DATASET2TYPE, so Waymo runs must still pass `-d kitti`. It picks
#   the metric set, and under --baseline it also picks the config (hence the data);
#   on the normal path the data follows the checkpoint's own config. Both table
#   entries are currently identical, so outside --baseline the choice is a no-op.
#
# Output:  <STAGED_DIR>/samples/<global_step>/<timestamp>/
#            img/    PNG range images
#            pcd/    samples_XXXXXX.txt  (generated)  +  inputs_XXXXXX.txt  (paired GT)
#            numpy/  samples.pcd
# Next:    symlink that directory into ../rectification/data/ (see its README).
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET="${1:?usage: sample_cond.sh STAGED_DIR_OR_CKPT [SPLIT] [METRIC_SET]}"
SPLIT="${2:-both}"
DSET="${3:-kitti}"

case "${SPLIT}" in train|validation|both) ;; *)
  echo "SPLIT must be 'train', 'validation' or 'both', got '${SPLIT}'"; exit 1 ;; esac

# --- auto-stage -------------------------------------------------------------
# sample_cond.py needs config.yaml and model.ckpt side by side. If handed a raw
# training logdir (checkpoints/ + configs/, no config.yaml), stage it first
# instead of failing on a missing config.yaml.
if [ -d "${TARGET}" ] && [ ! -f "${TARGET}/config.yaml" ]; then
  if [ -d "${TARGET}/checkpoints" ] && [ -d "${TARGET}/configs" ]; then
    STAGED="${TARGET}/${STAGE_NAME:-config_w_weights}"
    if [ -f "${STAGED}/config.yaml" ] && [ -f "${STAGED}/model.ckpt" ]; then
      echo ">>> reusing staged checkpoint: ${STAGED}"
    else
      echo ">>> ${TARGET} is a raw training logdir; auto-staging ${STAGE_CKPT:-last.ckpt}"
      bash "$(dirname "$0")/stage_checkpoint.sh" \
           "${TARGET}" "${STAGE_CKPT:-last.ckpt}" "${STAGE_NAME:-config_w_weights}"
    fi
    TARGET="${STAGED}"
  else
    echo "${TARGET} has neither config.yaml nor checkpoints/+configs/ - cannot stage"; exit 1
  fi
fi

case "${SPLIT}" in both) SPLITS=(train validation) ;; *) SPLITS=("${SPLIT}") ;; esac

SAMPLE_ROOT="${TARGET%/model.ckpt}/samples"

for sp in "${SPLITS[@]}"; do
  echo ">>> sampling: ${TARGET}  split=${sp}  metrics=${DSET}"
  python scripts/sample_cond.py -d "${DSET}" -r "${TARGET}" --split "${sp}" --eval
done

# sample_cond.py names each output directory <timestamp>_<split>, so the split is
# readable straight off the path.
echo
echo ">>> done. Symlink these into ../rectification/data/ (names differ per dataset -"
echo "    check train_data_root / val_data_root in the config you intend to run):"
for sp in "${SPLITS[@]}"; do
  d="$(ls -1dt "${SAMPLE_ROOT}"/*/*_"${sp}" 2>/dev/null | head -1 || true)"
  [ -n "${d}" ] && printf "      %-11s %s\n" "${sp}:" "$(cd "${d}" && pwd)"
done

# previously run:
#   python scripts/sample_cond.py -d kitti    -r <CKPT> --split validation --eval
#   python scripts/sample_cond.py -d nuscenes -r <CKPT> --split train      --eval
#   python scripts/sample_cond.py -d kitti    -r models/lidm/kitti/sem2lidar/model.ckpt --eval   # official weights
#   # re-evaluate an existing sample dump without re-sampling:
#   python scripts/sample_cond.py -d kitti -f <LOGDIR>/samples/<step>/<ts>/numpy/samples.pcd --eval
