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
# from lidm.eval.eval_utils import evaluate

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
        "-n",
        "--filename",
        type=str,
        help="filename to store the resulting image gradient lengths",
        required=True
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="only the first num_samples number of samples will be evaluated to save memory",
        default=100
    )
    parser.add_argument(
        "--num_grad_per_sample",
        type=int,
        help="reduce number of gradients to be recorded to save memory",
        default=100
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


def pcd2range_w_mask(pcd, size, fov, depth_range, remission=None, labels=None, lossless=True, **kwargs):
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    if lossless:
        # When proj_x or proj_y values are closely aligned to integers, rounding is very unstable.
        # if we are going to convert between RV and PCD multiple times, this is necessary.
        proj_x += 0.5
        proj_y += 0.5

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    valid_mask = np.logical_and(proj_range > 0, proj_range < depth_range[1])

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature, valid_mask


def calc_img_grad(img, valid_mask, horizontal_cyclic=True):
    # calculate image gradient
    img = img.astype(np.float32)
    H, W = img.shape

    # calculate gradients
    valid_x = np.logical_and(valid_mask[:, :W-1], valid_mask[:, 1:])
    valid_y = np.logical_and(valid_mask[:H-1, :], valid_mask[1:, :])
    grad_x = np.abs(np.diff(img, axis=1))
    grad_y = np.abs(np.diff(img, axis=0))

    # add cyclic gradient
    if horizontal_cyclic:
        valid_x_border = np.logical_and(valid_mask[:, :1], valid_mask[:, -1:])
        valid_x = np.concatenate([valid_x_border, valid_x], axis=1)
        grad_x = np.abs(np.diff(np.concatenate([img[:, -1:], img], axis=1), axis=1))

    # apply valid mask
    grad_x[~valid_x] = 0
    grad_y[~valid_y] = 0

    lengths = np.concatenate([grad_x[valid_x].flatten(), grad_y[valid_y].flatten()], axis=0)

    return grad_x, grad_y, lengths


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    filenames = sorted(os.listdir(opt.file))
    all_gt = []
    all_diff_output = []
    all_refined_diff_output = []
    if not opt.eval_refined_coords:
        print("Evaluating original, unrefined coordinates. This results should be the same for any logs.", flush=True)
    
    assert opt.dataset in ['nuscenes', 'kitti', 'waymo', 'kitti360']
    opt.base = [f'models/baseline/other_baselines/{opt.dataset}.yaml']
    
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    proj_params = config['data']['params']['dataset']

    if opt.eval_with_original_GT:
        data_config = config['data']['params'][opt.split]
        data_config['params'].update({'dataset_config': proj_params,
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True,
                                      'max_objects_per_image': 5})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=40, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=test_collate_fn)
        all_gt = []
        print(f"Loading original GT from {opt.split} split", flush=True)
        i = 0
        for batch in tqdm(dataloader, desc="Loading Batches (original GT)"):
            if i >= opt.num_samples:
                break
            all_gt.extend(batch['reproj'])
            i += 1
        
    for i, filename in enumerate(filenames):
        if opt.eval and i >= opt.num_samples:
            break
        filepath = os.path.join(opt.file, filename)
        data_dict = np.load(filepath, allow_pickle=True)
        in_xyz = data_dict["target"]
        rec_xyz = data_dict["input"]
        output_offsets = data_dict["output"]
        offsets = data_dict["label"]

        corrected_xyz = rec_xyz + rec_xyz / np.linalg.norm(rec_xyz, axis=-1, keepdims=True) * output_offsets[:, None]
        if not opt.eval_with_original_GT:
            all_gt.append(in_xyz)
            
        all_refined_diff_output.append(corrected_xyz)
        all_diff_output.append(rec_xyz)
    print("loading completed.", flush=True)

    # evaluation
    if opt.eval:
        if len(all_gt) < opt.num_samples:
            print("Required {} samples, but only {} samples found.".format(opt.num_samples, len(all_gt)))
        all_image_grad_seqs = {}
        for pcd_sequence, key in zip([all_gt, all_diff_output, all_refined_diff_output], ["gt", 'diff', 'refined_diff']):
            image_grad_seq = []
            print(f"processing {key} ...", flush=True)
            for i, pcd in enumerate(pcd_sequence):
                print(f"sample {i}", flush=True)
                if i >= opt.num_samples:
                    break
                rv_image, _, mask = pcd2range_w_mask(pcd, lossless=True, **proj_params)
                _, _, lengths = calc_img_grad(rv_image, mask, horizontal_cyclic=True)
                lengths = np.random.choice(lengths, size=opt.num_grad_per_sample, replace=False)
                image_grad_seq.extend(lengths)
            all_image_grad_seqs[key] = image_grad_seq

        np.savez(opt.filename, grads=all_image_grad_seqs)
        print(f"Saved image gradient lengths to {opt.filename}", flush=True)
