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
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0.0
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="PointOffsetRegressor",
    backbone=dict(
        type="ImageModelWrapper",
        backbone=dict(
            type="UNetModel",
            image_size = [64, 1024],
            in_channels = 1,
            out_channels = 1,
            model_channels = 128,
            attention_resolutions = [], # attention is disabled in model code due to complexity. Need vast modification to enable it.
            num_res_blocks = 2,
            channel_mult = [1, 2, 4],
            num_head_channels = 32,
            lib_name = 'lidm', # this only enables curcular convolution
        ),
    ),
    criteria=[
        # dict(type="MSELoss"),
        dict(type="WelshLoss", sigma=0.5),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=1e-6, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "SemanticKITTIGen2RealImageDataset"
train_data_root = "data/lidar-diffusion-waymo-semseg-cond-generated-trainingSet"
val_data_root = "data/lidar-diffusion-waymo-semseg-cond-generated"

pcd_to_range_config = dict(
    size = [64, 1024],
    fov = [ 2.2425447, -17.660652 ],
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
    is_image_model=True,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=train_data_root,
        with_sem_input=False,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("input", "label", "target", "color_real"),
            ),
        ],
        test_mode=False,
        pcd_to_range_config = pcd_to_range_config,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=val_data_root,
        with_sem_input=False,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("input", "label", "target", "color_real"),
            ),
        ],
        test_mode=False,
        pcd_to_range_config = pcd_to_range_config,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=val_data_root,
        with_sem_input=False,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("input", "label", "target", "color_real", "idx"),
            ),
        ],
        test_mode=False,
        pcd_to_range_config = pcd_to_range_config,
    ),
)
