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
batch_size = 16  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="PointOffsetRegressor",
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(3, 64, 128, 256),
        dec_num_head=(1, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
    ),
    criteria=[
        # dict(type="MSELoss"),
        dict(type="WelshLoss", sigma=2.0),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "SemanticKITTIGen2RealDataset"
train_data_root = "data/lidar-diffusion-kitti-semseg-cond-generated-trainingSet"
val_data_root = "data/lidar-diffusion-kitti-semseg-cond-generated"

pcd_to_range_config = dict(
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
        pcd_to_range_config = pcd_to_range_config,
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
