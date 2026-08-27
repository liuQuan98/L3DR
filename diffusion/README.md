# L3DR Stage 1: LiDAR Diffusion & Evaluation

Range-view LiDAR diffusion (LiDM) plus the evaluation toolbox. This half does three jobs:

1. **Trains** the two-stage generator: a VQ-VAE range-image compressor, then a latent
   diffusion model over it (unconditional, semantic-map-conditioned, or camera-conditioned).
2. **Samples** paired generated / ground-truth point clouds as the training data of the
   Residual Regression Network in [`../rectification/`](../rectification/).
3. **Scores** everything including FRID / FSVD / FPVD / JSD / MMD. The
   metrics are for rectified point clouds coming back from stage 2.

Derived from [LiDAR-Diffusion](https://github.com/hancyran/LiDAR-Diffusion) (Ran et al.,
CVPR 2024). It's own README is preserved at
[`docs/UPSTREAM_LiDM_README.md`](docs/UPSTREAM_LiDM_README.md). Refer to this file for the model-zoo
download tables.

## Install

```bash
bash init/create_env.sh          # creates conda env `lidar_diffusion` (Python 3.10.11)
conda activate lidar_diffusion
```

PyTorch-Lightning is pinned to **1.4.2** and is not interchangeable with newer versions.
This environment is separate from the one `../rectification/` uses.

For the CD / EMD metrics, build the CUDA extensions once:

```bash
bash lidm/eval/compile.sh        # builds chamfer2D, chamfer3D, emd
```

FSVD and FPVD additionally need `torchsparse==1.4.0` and the Google sparsehash library
(`apt-get install libsparsehash-dev`); `create_env.sh` installs both.

Running `compile.sh` is optional — the extensions are JIT-compiled on first use if no
prebuilt copy is found. Building ahead of time just moves that cost out of your first
evaluation.

> **The evaluation dependencies are required for sampling, not only evaluation.**
> `scripts/sample_cond.py` imports `lidm.eval.eval_utils` at module level, so a missing
> `scikit-learn` fails the sampling run before a single frame is generated. If you built
> your environment before `create_env.sh` gained that dependency, install it:
> `pip install scikit-learn scipy tqdm joblib`.

## Data

Symlink raw datasets into `dataset/` (not committed — machine-specific). Notice that SemanticKITTI and KITTI360 are verbatim as downloaded files; nuScenes and Waymo needs convertion to Pointcept format. We refer readers to the conversion process for [nuScenes](https://github.com/pointcept/pointcept#nuscenes) and [Waymo](https://github.com/pointcept/pointcept#nuscenes). After data processing, symlink the datasets here:

```bash
mkdir -p dataset
ln -s /path/to/SemanticKITTI   dataset/SemanticKITTI
ln -s /path/to/KITTI-360       dataset/KITTI-360
ln -s /path/to/nuScenes        dataset/nuScenes-converted
ln -s /path/to/Waymo           dataset/Waymo-Pointcept-format
```

## Weights

**No weights are committed** — only the `config.yaml` scaffolding that tells you where
they go. This mirrors upstream.

- **LiDM + autoencoder checkpoints** → `models/…`. Download links are in the tables in
  [`docs/UPSTREAM_LiDM_README.md`](docs/UPSTREAM_LiDM_README.md). A config's
  `first_stage_config.params.ckpt_path` names the autoencoder it expects, e.g.
  `models/first_stage_models/kitti/f_c2_p4_wo_ls/model.ckpt`.
- **Evaluation backbones** (RangeNet++ for FRID, MinkowskiNet for FSVD, SPVCNN for FPVD)
  → `pretrained_weights/` (~760 MB). Links and per-metric tables are in
  [`lidm/eval/README.md`](lidm/eval/README.md). Unzip so the layout is
  `pretrained_weights/<dataset>/<model>/`, or repoint `DEFAULT_ROOT` in
  `lidm/eval/__init__.py`.

## Running the pipeline

`scripts/pipeline/` holds one script per stage. They are thin wrappers over `main.py`
and `scripts/*.py` — you can always call those directly — but they carry the defaults
used for the paper, and each keeps the historical invocations as `# previously run:`
comments.

A full semantic-conditioned KITTI run is four commands:

```bash
GPUS=0,1,2,3 bash scripts/pipeline/train_autoencoder.sh          # 1
GPUS=0,1,2,3 bash scripts/pipeline/train_diffusion.sh            # 2
bash scripts/pipeline/sample_cond.sh <LOGDIR>                    # 3
bash scripts/pipeline/eval_refined_coords.sh <RESULT_DIR> kitti  # 4
```

Each script takes its target as the first argument and falls back to the SemanticKITTI
config when omitted. `<LOGDIR>` is the run step 2 created,
`logs/<dataset>/<timestamp>_<config>/`.

### 1–2. Training

Train the autoencoder, point the diffusion config's
`first_stage_config.params.ckpt_path` at the resulting checkpoint, then train the
diffusion model. Pass a config to either script to run a different dataset:

```bash
GPUS=1,2,3 bash scripts/pipeline/train_diffusion.sh configs/lidar_diffusion/waymo/semantic_cond_c2_p4_wo_ls.yaml

# resume, via main.py directly
python main.py -b <config> -t --gpus 0, -r <logdir-or-ckpt>
```

Available config families (`configs/lidar_diffusion/<dataset>/`):

| Dataset | Conditioning |
|---|---|
| `semantickitti/` | `semantic_cond`, `semantic_uncond…w_seg_input`, `clip_cond`, `semantic_cond_point-only` |
| `nuscenes/`, `waymo/` | `semantic_cond` |
| `kitti360/` | `uncond`, `image_cond` (camera-to-LiDAR) |

Note: The LiDAR-Diffusion original weights produce much better training data, so we would use that to generate RRN training data on KITTI instead.

#### How long training runs

Both stages stop at a fixed step budget set as `lightning.trainer.max_steps` in every
config — **40,000** steps for autoencoders and **10,000** for diffusion models. These are
the LiDM author's figures from [`DESIGN.md`](DESIGN.md) ("trained with 40k steps" /
"trained with 10k steps", measured on 4× 3090).

This matters because **PyTorch Lightning does not stop on its own**. With no `max_steps`
and no `max_epochs` it falls back to `max_epochs=1000`, which on SemanticKITTI
(~19k frames at `batch_size: 16`) is roughly 1.2M steps — training that never
realistically ends. There is no early stopping anywhere in `main.py`.

Treat the budgets as a starting point rather than a target: they come from the author's
compression *ablation*, not necessarily from the runs that produced the released
checkpoints. Raise them in the config, or override per run without editing anything:

```bash
python main.py -b <config> -t --gpus 0,1,2,3 --max_steps 80000
```

Note the autoencoder configs also set `accumulate_grad_batches: 2`, so 40k optimizer
steps is 80k forward batches.

Interrupting training is safe, and picking an earlier checkpoint is expected —
`main.py` keeps `checkpoints/last.ckpt` current, retains the best three by the config's
`monitor` (`val/rec_loss` for autoencoders, `val/loss_simple_ema` for diffusion), and
writes an unpruned checkpoint every 10,000 steps into
`checkpoints/trainstep_checkpoints/`. `stage_checkpoint.sh` takes a `CKPT_NAME` so you
can sample from whichever one you choose.

**nuScenes note.** Its RV image is over-provisioned to `[64, 1024]` rather than its
native 32 rows — a `[32, 1024]` image diverges during training. See the paper, §5.2.

### 3. Sampling — producing stage-2 training data

`sample_cond.sh` stages the checkpoint and samples both splits; pass `train` or
`validation` as a second argument to run just one. For unconditional models:

```bash
bash scripts/pipeline/sample_uncond.sh <CKPT> 2000
```

Each split run writes its own directory, named `<timestamp>_<split>`, so the two runs
never overwrite each other and the split is readable straight off the path.
`sample_cond.sh` prints both paths at the end, ready to symlink into
`../rectification/data/`.

Output lands in `<STAGED_DIR>/samples/<global_step>/<timestamp>_<split>/` — i.e. beside
whichever checkpoint you sampled from, which for the released weights means inside
`models/`. It is gitignored, but it is not small: ~260 MB per 70 frames.

Contents:

```
img/     PNG range images
pcd/     samples_XXXXXX.txt  ← generated point cloud   (xyz + rgb, %.3f)
         inputs_XXXXXX.txt   ← paired ground truth
numpy/   samples.pcd         ← joblib list, used by --eval
```

`../rectification/` consumes `pcd/` directly and pairs each `samples_*` with its
`inputs_*` partner **by filename**. Do not rename these outputs.

### 4. Evaluation

For generation quality on its own, the sampling scripts already pass `--eval`.

To score rectified point clouds coming back from stage 2:

```bash
bash scripts/pipeline/eval_refined_coords.sh <RESULT_DIR> kitti [MODE]
```

`MODE` is `baseline`, `refined`, or `both` (default) — `both` scores the unrectified and
rectified clouds from the same `.npz` files, which is how a baseline/rectified table row
pair is produced. Two environment variables adjust it:

- `ORIGINAL_GT=1` loads ground truth from the **raw dataset** instead of from the
  `.npz`, via `models/baseline/other_baselines/<dataset>.yaml`, so it needs the
  `dataset/` symlinks. The distinction matters because RVP is not lossless in all
  cases: this compares against unresampled ground truth, making it the more
  trustworthy of the two comparisons.
- `NUM_FILES=N` caps how many frames are scored.
- `METRICS=jsd,mmd` restricts which metrics run. The default is
  `frid,fsvd,fpvd,jsd,mmd`; `frid`/`fsvd`/`fpvd` each load a pretrained backbone from
  `pretrained_weights/` and are the slow ones, and `mmd` is an O(n²) pairwise chamfer,
  so `METRICS=jsd` is the quick sanity check.

**GPU memory.** The perceptual metrics batch for a ~24 GB card. If the GPU is shared you
will hit `torch.OutOfMemoryError` partway through — lower `MODAL2BATCHSIZE` in
[`lidm/eval/__init__.py`](lidm/eval/__init__.py), cap `NUM_FILES`, or drop to
`METRICS=jsd,mmd`, which needs no backbone.

Paper Figure 2 (RV image-gradient distribution, GT vs diffused vs rectified) has no
wrapper:

```bash
python scripts/eval_img_grad_distribution.py --filename <out>.npz
```

### Checkpoint staging

Sampling needs `config.yaml` and `model.ckpt` in one directory. `sample_cond.sh` builds
that for you from a training log directory and reuses it on later runs, so normally you
never invoke this. To stage a specific checkpoint explicitly:

```bash
bash scripts/pipeline/stage_checkpoint.sh <LOGDIR> [CKPT_NAME] [OUT_NAME]
```

| parameter | default | change it to |
|---|---|---|
| `LOGDIR` | — | the training run under `logs/<dataset>/<timestamp>_<config>/` |
| `CKPT_NAME` | `last.ckpt` | any file in `<LOGDIR>/checkpoints/`, e.g. `epoch=000052.ckpt` |
| `OUT_NAME` | `config_w_weights` | another name, e.g. `config_w_weights_final`, to keep several staged checkpoints side by side |

The same two choices are available to `sample_cond.sh` as `STAGE_CKPT=` and
`STAGE_NAME=`, so you can sample from a specific epoch in one command:

```bash
STAGE_CKPT=epoch=000052.ckpt STAGE_NAME=config_w_weights_final \
  bash scripts/pipeline/sample_cond.sh <LOGDIR>
```
