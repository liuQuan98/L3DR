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
        dict(type="WelshLoss", sigma=0.2),
    ],
)

# scheduler settings
epoch = 1000
eval_epoch = 1000
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
train_data_root = "data/gs-lidar_reconstruction_log"
val_data_root = "data/gs-lidar_reconstruction_log"

pcd_to_range_config = dict(
    size = [66, 1030],
    fov = [ 2.0, -24.9 ],
    depth_range = [ 1.0,80.0 ],
    depth_scale = 80,  # np.log2(depth_max + 1)
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
    train_w_val_data = True,  # whether to train with validation data (4/50 frames, insdead of all 50/50 frames)
)

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        loop=20,
        data_root=train_data_root,
        transform=[
            dict(
                type="Copy",
                keys_dict=dict(coord="origin_coord")
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "label", "coord_real", "origin_coord"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "label", "coord_real", "origin_coord"),
                feat_keys=["coord",],
            ),
        ],
        test_mode=False,
        pcd_to_range_config=pcd_to_range_config,
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
        pcd_to_range_config = pcd_to_range_config,
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
        pcd_to_range_config = pcd_to_range_config,
    ),
)
