_base_ = ["../_base_/default_runtime.py"]

test = dict(type="RegressionTester", verbose=True)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="RegressionEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

# misc custom setting
batch_size = 60  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="PointOffsetRegressor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=3,
        channels=(32, 64, 128, 256, 256, 128, 64, 64),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        # dict(type="MSELoss"),
        dict(type="WelshLoss", sigma=0.5),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "SemanticKITTIGen2RealDataset"
train_data_root = "data/lidar-diffusion-kitti-semseg-cond-generated-trainingSet"
# val_data_root = "data/r2dm-kitti360-unconditional-generated"
val_data_root = "data/r2dm-kitti360-unconditional-generated-ddim50steps"

# Cross-sensor generalization: train on kitti, evaluate on kitti360.
# Each split needs its OWN range-view projection - one shared config
# cannot describe two different LiDAR geometries.
train_pcd_to_range_config = dict(
    size = [64, 1024],
    fov = [ 3, -25 ],
    depth_range = [ 1.0,56.0 ],
    depth_scale = 56,  # np.log2(depth_max + 1)
    log_scale = False,
    x_range = [ -50.0, 50.0 ],
    y_range = [ -50.0, 50.0 ],
    z_range = [ -3.0, 1.0 ],
    resolution = 1,
    num_channels = 1,
    num_cats = 10,
    num_views = 2,
    num_sem_cats = 19,
    filtered_map_cats = [ ],
)

val_pcd_to_range_config = dict(
    size = [64, 1024],
    fov = [ 3, -25 ],
    depth_range = [ 1.0,56.0 ],
    depth_scale = 56,  # np.log2(depth_max + 1)
    log_scale = False,
    x_range = [ -50.0, 50.0 ],
    y_range = [ -50.0, 50.0 ],
    z_range = [ -3.0, 1.0 ],
    resolution = 1,
    num_channels = 1,
    num_cats = 10,
    num_views = 2,
    num_sem_cats = 19,
    filtered_map_cats = [ ],
)

# top-level alias: tester's image-model branch reads `pcd_to_range_config`
pcd_to_range_config = val_pcd_to_range_config

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=train_data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "label", "coord_real"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "label", "coord_real"),
                feat_keys=["coord",],
            ),
        ],
        test_mode=False,
        pcd_to_range_config = train_pcd_to_range_config,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=val_data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "label", "coord_real"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "label", "coord_real"),
                feat_keys=["coord",],
            ),
        ],
        test_mode=False,
        pcd_to_range_config = val_pcd_to_range_config,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=val_data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "label", "coord_real", "color_real"),
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "label", "coord_real", "inverse", "idx", "color_real"),
                feat_keys=["coord",],
            ),
        ],
        test_mode=False,
        pcd_to_range_config = val_pcd_to_range_config,
    ),
)
