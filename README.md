# L3DR: 3D-aware LiDAR Diffusion and Rectification

Official implementation of **"L3DR: 3D-aware LiDAR Diffusion and Rectification"**
Quan Liu, Xiaoqin Zhang, Ling Shao, Shijian Lu.

Paper link: [CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_L3DR_3D-aware_LiDAR_Diffusion_and_Rectification_CVPR_2026_paper.html), [arxiv](https://arxiv.org/abs/2602.19064)

Range-view (RV) LiDAR diffusion produces 2D-photorealistic range images, but
back-projecting them to 3D exposes geometry artifacts: *depth bleeding* at object
borders and *wavy surfaces* on flat geometry. L3DR adds a 3D **Residual Regression
Network (RRN)** that predicts a per-point radial offset and cancels those artifacts,
supervised by a **Welsch Loss** that ignores high-bias regions in the training pairs.

The method is diffusion-agnostic: an RRN trained on semantic-conditioned LiDM output
transfers to unseen unconditional diffusion models (R2DM, unconditional LiDM) at
~20 ms and ~38 M parameters of overhead.

## Bugfix and notes

Preparing this release surfaced a bug in the research code: the nuScenes and Waymo
RRN configs used SemanticKITTI's range-view field of view, `[3, -25]`,
causing the RRN to only correct a part of the full FoV on nuScenes and Waymo. SemanticKITTI and
KITTI360 performances are unaffected. **The nuScenes and Waymo numbers in the current paper were therefore produced with the
deteriorated setting, and this code will likely have better performance.**

Updated results will be posted to arxiv recently.

## Repository layout

This repository holds **both halves of the pipeline**. They are separate codebases
with separate conda environments and cannot be merged:

| Folder | Role | Base |
|---|---|---|
| [`diffusion/`](diffusion/) | **Stage 1** — trains the LiDAR diffusion model, samples generated/GT point-cloud pairs, and computes *all* reported metrics | [LiDAR-Diffusion / LiDM](https://github.com/hancyran/LiDAR-Diffusion) |
| [`rectification/`](rectification/) | **Stage 2** — trains and runs the Residual Regression Network | [Pointcept](https://github.com/Pointcept/Pointcept) |

> **Two environments.** `diffusion/` needs Python 3.10 + PyTorch-Lightning 1.4.2;
> `rectification/` needs the Pointcept stack (spconv, torch-scatter, optionally
> FlashAttention). Do not try to share one env — the pinned Lightning version alone
> makes them incompatible. Each folder's README has its own install section.

## The pipeline

The two halves hand off through the filesystem, twice. Run in this order:

```
                      diffusion/                                  rectification/
  ┌──────────────────────────────────────────┐        ┌──────────────────────────────┐
  │ 1. train VQ-VAE      main.py             │        │                              │
  │ 2. train LiDM        main.py             │        │                              │
  │ 3. sample            scripts/sample_cond │───(a)─▶│ 4. train RRN   scripts/train │
  │                                          │        │ 5. test RRN    scripts/test  │
  │ 6. score      scripts/eval_refined_coords│◀──(b)──│                              │
  └──────────────────────────────────────────┘        └──────────────────────────────┘
```

**(a) Generated data → RRN.** `sample_cond.py` writes, per run,
`<logdir>/samples/<step>/<timestamp>/pcd/` containing **both** `samples_XXXXXX.txt`
(generated) and `inputs_XXXXXX.txt` (the paired ground truth). `rectification/` reaches
these through a symlink in `rectification/data/`, globs `pcd/samples*.txt`, and recovers
each GT partner by substituting `samples`→`inputs` in the filename.
**That filename pairing is the contract between the two halves** — renaming sampler
outputs silently breaks RRN training.

**(b) Rectified output → metrics.** RRN testing writes
`rectification/exp/<d>/<n>/result_<split>/*.npz` with keys `input` (generated xyz),
`target` (GT xyz), `label` (GT radial offset), `output` (predicted offset).
`eval_refined_coords.py` reads those, applies

```
P_rectified = P_gen + P_gen/‖P_gen‖ · predicted_offset
```

and reports FRID / FSVD / FPVD / JSD / MMD. **`rectification/` implements none of these
metrics** — every number in the paper's tables is produced in `diffusion/`.

## Quick start

```bash
# stage 1
cd diffusion && bash init/create_env.sh && conda activate lidar_diffusion
python main.py -b configs/autoencoder/semantickitti/autoencoder_c2_p4_wo_ls.yaml -t --gpus 0,1,2,3
python main.py -b configs/lidar_diffusion/semantickitti/semantic_cond_c2_p4_wo_ls.yaml -t --gpus 0,1,2,3
bash scripts/pipeline/sample_cond.sh <logdir>        # samples BOTH splits

# stage 2  (separate env)
# The RRN trains on the train-split samples and is tested on the validation-split
# samples, so BOTH need a symlink. Each split run wrote its own timestamped dir.
cd ../rectification && conda activate pointcept
ln -s <train-split sample dir> data/lidar-diffusion-kitti-semseg-cond-generated-trainingSet
ln -s <val-split sample dir>   data/lidar-diffusion-kitti-semseg-cond-generated
sh scripts/train.sh -g 4 -d generated_to_real_refinement \
   -c offset-prediction-spunet-0-base-from-lidar-diffusion-kitti-semseg_to-kitti-real_regressVector -n my-run
sh scripts/test.sh  -g 1 -d generated_to_real_refinement -n my-run -w exp/generated_to_real_refinement/my-run/model/model_best

# back to stage 1 for metrics
cd ../diffusion
python scripts/eval_refined_coords.py -d kitti --eval --eval_refined_coords \
   --file ../rectification/exp/generated_to_real_refinement/my-run/result_val
```

Drop `--eval_refined_coords` to score the *unrectified* baseline from the same files —
that is how a baseline/rectified table row pair is produced.

## Acknowledgement

Built on [LiDAR-Diffusion (LiDM)](https://github.com/hancyran/LiDAR-Diffusion) by Ran et al.
and [Pointcept](https://github.com/Pointcept/Pointcept). Please cite those excellent papers
when citing L3DR.
