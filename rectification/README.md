# L3DR — Stage 2: Residual Regression Network

Trains and runs the **Residual Regression Network (RRN)**: a 3D backbone that takes a
diffusion-generated point cloud and predicts a per-point offset which cancels range-view
artifacts — depth bleeding at borders, waviness on flat surfaces.

Derived from [Pointcept](https://github.com/Pointcept/Pointcept), carved down to the
L3DR pipeline. Requires generated data produced by [`../diffusion/`](../diffusion/),
and produces `.npz` files that `../diffusion/` scores.

## How the RRN works

For a generated cloud `P_gen`, a backbone `F` predicts a 3-vector per point. That vector
is **projected onto each point's radial direction**, so the correction only moves points
along the sensor ray — which is how range-image depth error actually behaves:

```
Ô = P_gen · diag(P_gen Fᵀ) / ‖P_gen‖        P_rectified = P_gen + Ô
```

Supervision is a **Welsch Loss**, `ψ_ν(x) = 1 − exp(−x²/2ν²)`. Its bounded influence
makes it ignore the high-bias regions scattered through the training pairs (depth-shifted
walls, hallucinated chunks) and concentrate on the small, high-variance RV artifacts.
`ν` is the `sigma` argument of `WelshLoss`; `configs/gen2real_refine_ablation/` sweeps it.

The regression target is built in range-view: the dataset projects both the generated
and the real cloud to range images, takes `offset_label = img_real − img_gen`, then
back-projects with a mask intersection so the two clouds correspond point-for-point.
The 3D loss and the paper's `RVP(·)`-wrapped Eq. 6 are therefore the same quantity.

## Install

Use the Pointcept stack — **a different conda env from `../diffusion/`**:

```bash
conda create -n pointcept python=3.8 -y && conda activate pointcept
conda install ninja -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric spconv-cu113 open3d opencv-python
```

The backbones L3DR uses (SpUNet, PTv3) need `spconv` and `torch-scatter`; PTv3 optionally
uses FlashAttention (`enable_flash=True` in its configs). The CUDA extensions in `libs/`
are only required by backbones L3DR does not use — build them only if you add one:

```bash
cd libs/pointops && python setup.py install && cd ../..
```

## Data

The RRN reads **generated/GT pairs produced by stage 1**, and it needs **two** of them:
it trains on the samples generated from the *training* split and is tested on the
samples generated from the *validation* split. Both need a symlink in `data/` — a
config's `train_data_root` and `val_data_root` point at different directories.

Stage 1's `sample_cond.sh` samples both splits in one call. Each run writes its own
directory under `<STAGED_DIR>/samples/<step>/`, named `<timestamp>_<split>` — so the two
never overwrite each other and you can tell them apart from the path alone:

```
samples/00015847/2024-10-22-09-11-33_train/
samples/00015847/2024-10-22-14-30-01_validation/
```

**The expected symlink names are not uniform across datasets** — nuScenes suffixes its
validation directory, KITTI and Waymo leave it bare:

| Dataset | `train_data_root` | `val_data_root` |
|---|---|---|
| SemanticKITTI | `data/lidar-diffusion-kitti-semseg-cond-generated-trainingSet` | `data/lidar-diffusion-kitti-semseg-cond-generated` |
| nuScenes | `data/lidar-diffusion-nuscenes-semseg-cond-generated-trainingSet` | `data/lidar-diffusion-nuscenes-semseg-cond-generated-validationSet` |
| Waymo | `data/lidar-diffusion-waymo-semseg-cond-generated-trainingSet` | `data/lidar-diffusion-waymo-semseg-cond-generated` |

So for SemanticKITTI:

```bash
mkdir -p data
ln -s ../../diffusion/logs/<run>/config_w_weights/samples/<step>/<timestamp>_train \
      data/lidar-diffusion-kitti-semseg-cond-generated-trainingSet
ln -s ../../diffusion/logs/<run>/config_w_weights/samples/<step>/<timestamp>_validation \
      data/lidar-diffusion-kitti-semseg-cond-generated
```

Check the names in the config you intend to run before creating the links. The diffusion-agnostic inference
configs (`*_R2DM_GeneratedInference`, `*_KITTI360Inference`, `*_KITTIUnconditionalInference`)
point `val_data_root` at other generated sources — see each config.

The dataset globs `pcd/samples*.txt` under each root and pairs every file with its
`inputs_*.txt` partner by filename.

## Training and testing

```bash
# -g GPUs, -d config family, -c config basename (no .py), -n experiment name
sh scripts/train.sh -g 4 -d generated_to_real_refinement \
   -c offset-prediction-spunet-0-base-from-lidar-diffusion-kitti-semseg_to-kitti-real_regressVector \
   -n my-run

sh scripts/train.sh -g 4 -d <family> -c <config> -n <exp> -r true       # resume

sh scripts/test.sh  -g 1 -d <family> -n <exp> -w exp/<family>/<exp>/model/model_best
```

Two things about `train.sh` that surprise people:

- It **snapshots** `pointcept/`, `tools/` and `scripts/` into `exp/<family>/<exp>/code/`
  and runs *that copy*. Edits to the working tree do not affect a running or resumed job.
- On `-r true` it reloads `exp/<family>/<exp>/config.py`, so the `-c` argument is ignored.

`test.sh` deliberately does the opposite — it runs the working tree. Note `-w` is
interpolated as `"<value>".pth`, so pass a path without the extension, as above.

## Output

Testing writes one `.npz` per frame to `exp/<family>/<exp>/result_<split>/`:

| key | meaning |
|---|---|
| `input` | generated point cloud `P_gen` |
| `target` | ground-truth point cloud |
| `label` | ground-truth radial offset |
| `output` | **predicted** radial offset |
| `inverse`, `color`, `loss` | voxel-inverse map, semantic colors, per-frame loss |

Existing `.npz` files are reused rather than recomputed — delete the directory to
force a fresh run after retraining.

Metrics (FRID/FSVD/FPVD/JSD/MMD) are not implemented here; feed this directory to
`../diffusion/scripts/eval_refined_coords.py`.

## Configs

Configs are Python, merged over `configs/_base_/default_runtime.py` via `_base_`.
Every runtime knob — `epoch`, `batch_size`, `hooks`, `train`, `test` — is a module-level
variable; `--options k=v` overrides them from the CLI.

| Family | Purpose |
|---|---|
| `generated_to_real_refinement/` | main results — per dataset, per backbone, ± semantic input |
| `gen2real_refine_ablation/` | Welsch `sigma` (ν) sweep |
| `gen2real_refine_generalization/` | cross-sensor transfer; diffusion-agnostic inference (R2DM, unconditional, KITTI360, GS-LiDAR) |

Naming: `offset-prediction-<backbone>-…-<train data>_to-<target>_regress{Vector,Scalar}[_variant]`.
`WithSemInput` feeds the semantic color map alongside coordinates (`in_channels=6`).

Backbones: `SpUNet-v1m1` (paper default, 37.9 M), `PT-v3m1`, and a 2D `UNetModel`
control selected by `data.is_image_model = True` (paper Table 3).
