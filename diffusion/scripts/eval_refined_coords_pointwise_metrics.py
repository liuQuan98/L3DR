import math
import sys

sys.path.append('./')

import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import joblib

from omegaconf import OmegaConf
from PIL import Image

from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.utils.lidar_utils import range2pcd
from lidm.eval.eval_utils import evaluate

# remove annoying user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# DATASET2METRICS = {'kitti': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd'], 'nuscenes': ['fsvd', 'fpvd']}
# DATASET2TYPE = {'kitti': '64', 'nuscenes': '32'}

# custom nuscenes val metrics with 64 lasers
DATASET2METRICS = {'kitti': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd'], 'nuscenes': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd']}
DATASET2TYPE = {'kitti': '64', 'nuscenes': '64'}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default="none"
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--vanilla",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "-f",
        "--file",
        help="the file path of samples",
        default=None
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="the numpy file path",
        default=1000
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name [nuscenes, kitti]",
        required=True
    )
    parser.add_argument(
        "--baseline",
        default=False,
        action='store_true',
        help="baseline provided by other sources (default option is not baseline)?",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action='store_true',
        help="print status?",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action='store_true',
        help="evaluation results?",
    )
    parser.add_argument(
        "--eval_refined_coords",
        default=False,
        action='store_true',
        help="whether to eval refined voxelized coords from pointcept, or the original coords voxelized",
    )
    parser.add_argument(
        "--split",
        default='validation',
        help="which split to use during sampling",
    )
    parser.add_argument(
        "--eval_with_original_GT",
        default=False,
        action='store_true',
        help="whether to use lidar-diffusion original GT for evaluation, defaults to false which uses pointcept GT",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        help="the number of files to evaluate",
        default=-1,
    )
    return parser


def test_collate_fn(data):
    output = {}
    keys = data[0].keys()
    for k in keys:
        v = [d[k] for d in data]
        if k not in ['reproj', 'raw']:
            v = torch.from_numpy(np.stack(v, 0))
        else:
            v = [d[k] for d in data]
        output[k] = v
    return output


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    filenames = sorted(os.listdir(opt.file))
    if opt.num_files > 0:
        filenames = filenames[:opt.num_files]
    all_gt = []
    all_samples = []
    if not opt.eval_refined_coords:
        print("Evaluating original, unrefined coordinates. This results should be the same for any logs.", flush=True)
    
    if opt.eval_with_original_GT:
        assert opt.dataset in ['nuscenes', 'kitti', 'waymo', 'kitti360']

        opt.base = [f'models/baseline/other_baselines/{opt.dataset}.yaml']
        
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        data_config = config['data']['params'][opt.split]
        data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True,
                                      'max_objects_per_image': 5})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=40, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=test_collate_fn)
        all_gt = []
        print(f"Loading original GT from {opt.split} split", flush=True)
        for batch in tqdm(dataloader, desc="Loading Batches (conditional)"):
            all_gt.extend(batch['reproj'])
        
    for filename in filenames:
        filepath = os.path.join(opt.file, filename)
        data_dict = np.load(filepath, allow_pickle=True)
        in_xyz = data_dict["target"]
        rec_xyz = data_dict["input"]
        output_offsets = data_dict["output"]
        offsets = data_dict["label"]

        corrected_xyz = rec_xyz + rec_xyz / np.linalg.norm(rec_xyz, axis=-1, keepdims=True) * output_offsets[:, None]
        if not opt.eval_with_original_GT:
            all_gt.append(in_xyz)
        if opt.eval_refined_coords:
            all_samples.append(corrected_xyz)
        else:
            all_samples.append(rec_xyz)

    # evaluation
    if opt.eval:
        # metrics = ['cd', 'emd', 'edgermse', 'regularity']
        metrics = ['regularity']
        data_type = "64"
        evaluate(all_gt, all_samples, metrics, data_type)
