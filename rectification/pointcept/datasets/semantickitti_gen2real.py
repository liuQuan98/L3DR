"""
Generated Semantic KITTI data from Lidar-Diffusion (https://github.com/hancyran/LiDAR-Diffusion)

Author: Quan Liu (quan.liu@ntu.edu.sg)
Adopted from the original PTV3 semantickitti implementation, which is written by: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
"""

import os
import numpy as np
import glob
import cv2
import copy
import open3d as o3d

from .builder import DATASETS
from .defaults import DefaultDataset
from .lidar_utils import range2pcd, range2pcd_w_mask, pcd2range


@DATASETS.register_module()
class SemanticKITTIGen2RealDataset(DefaultDataset):    
    def __init__(
        self,
        split="train",
        data_root="data/lidar-diffusion-official-semseg-log",
        pred_root=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        pcd_to_range_config=None,
    ):
        self.pred_root = pred_root
        # delete these configuration as it is not used in this dataset.
        # depth scale and log scale are used to compress depth into -1 to 1 range, where we need not use that
        # as we convert pcd to range image and back to pcd immediately, without intermediate diffusion.
        pcd_to_range_config.pop("depth_scale")
        pcd_to_range_config.pop("log_scale")
        self.pcd_to_range_config = pcd_to_range_config
        assert test_mode == False, "Test mode is not supported in generated data now."
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        # assert self.split in ["train", "val"], '''Only training is supported in generated data now. 
        #                                           Test/val may be introduced when we generate more data by ourselves.'''
        data_list = []
        seq_folder = os.path.join(self.data_root, "pcd")
        data_list = glob.glob(os.path.join(seq_folder, 'samples*.txt'))

        if len(data_list) == 0:
            print(f"Warning: No pcd/samples_XXXXXX.txt data found in {self.data_root}. Retrying with R2DM r2dm_samples_*.npz")
            data_list = glob.glob(os.path.join(self.data_root, '*.npz'))

        if len(data_list) == 0:
            data_phase = "train" if self.split == "train" and \
                                    not self.pcd_to_range_config.get("train_w_val_data", False) \
                                 else "test"
            print(f"Warning: No R2DM *.npz data found in {self.data_root}. Retrying with gs-lidar reconstruction data.")
            data_list = glob.glob(os.path.join(self.data_root, '*', 'eval', f'{data_phase}_refine_render', 'pred_*.ply'))

            if len(data_list) != 0:
                if self.pcd_to_range_config.get("train_w_val_data", False):
                    # if train with val data, we also include val data
                    data_list_val = data_list[::4]  # one scene generates 4 frames, every 4th is validation data
                    data_list_train = data_list[1::4] + data_list[2::4] + data_list[3::4]  # the rest are training data

                    if self.split == "train":
                        data_list = data_list_train
                    elif self.split == "val":
                        data_list = data_list_val

        if len(data_list) == 0:
            raise FileNotFoundError(f"No pcd/samples_XXXXXX.txt data found in {self.data_root}.")

        return data_list

    def rgb2segment(self, rgb):
        '''Convert LiDAR Diffusion label space (19-class mapped RGB) to Pointcept label space (19-class index)'''
        rgb = rgb*256 # convert 0-1 to 0-255
        dist = ((rgb[:, None, :] - self.LABEL2RGB[None, :, :])**2).sum(-1) # we assume GT class to have minimum eudlidean distance to mapped values on RGB space
        segment_w_ignore = np.argmin(dist, axis=-1).astype(np.int32)
        segment = segment_w_ignore - 1 # convert from {0:ignore, 1: car, ...} to {-1:ignore, 0:car, ...}
        segment[np.where(segment==-1)] = self.ignore_index # customizable ignore index
        return segment
    
    def process_rgb(self, rgb, skip_rescale=False):
        rgb = rgb / 255.
        if skip_rescale:
            rgb = np.clip(rgb, 0, 1)
        else:
            rgb = rgb * 2. - 1.
            rgb = np.clip(rgb, -1, 1)
        return rgb
        
    def get_data(self, idx):
        # image file names
        gen_data_path = self.data_list[idx % len(self.data_list)]

        if "r2dm" in gen_data_path[-28:]:
            # r2dm unconditional point cloud, other items are filed with dummy.
            real_data_path = gen_data_path # r2dm is unconditional. Does not have real_data_path
            xyz = np.load(gen_data_path, allow_pickle=True)["xyz"]
            scan_gen = np.concatenate([xyz, np.zeros_like(xyz)], axis=-1)
            scan_real = copy.deepcopy(scan_gen)
        elif gen_data_path.endswith(".ply"):
            # gs-lidar reconstruction data
            # in this mode, we apply refinement on the diffusion data log.
            pcd_rec = o3d.io.read_point_cloud(gen_data_path)
            gt_data_path = gen_data_path.replace("pred_", "gt_")
            pcd_gt = o3d.io.read_point_cloud(gt_data_path)
            scan_gen = np.asarray(pcd_rec.points).astype(np.float32)
            scan_real = np.asarray(pcd_gt.points).astype(np.float32)

            scan_gen = np.concatenate([scan_gen, np.zeros_like(scan_gen)], axis=-1)  # add dummy color
            scan_real = np.concatenate([scan_real, np.zeros_like(scan_real)], axis=-1)

            # swap x and y axis to match pointcept coordinate system
            # pointcept uses z as up axis, while gs-lidar uses y as up axis
            # so we swap x and y axis, and negate z axis
            scan_gen[:, [0, 1, 2]] = scan_gen[:, [2, 0, 1]]
            scan_real[:, [0, 1, 2]] = scan_real[:, [2, 0, 1]]

            scan_gen[:, 2] = -scan_gen[:, 2]
            scan_real[:, 2] = -scan_real[:, 2]
        else:
            # pcd/samples_XXXXXX.txt, paired with pcd/inputs_XXXXXX.txt by filename
            # in this mode, we apply refinement on the diffusion data log.
            # Substitute in the BASENAME only: the full path may contain "samples" again as a
            # directory component (logdir/samples/<step>/<ts>_<split>/pcd/samples_*.txt), and a
            # whole-string replace would rewrite that too and point at a directory that does
            # not exist. Symlinked data_roots hide this; absolute ones do not.
            _gen_dir, _gen_name = os.path.split(gen_data_path)
            real_data_path = os.path.join(_gen_dir, _gen_name.replace("samples", "inputs", 1))
            scan_gen = np.loadtxt(gen_data_path, dtype=np.float32).reshape(-1, 6)
            scan_real = np.loadtxt(real_data_path, dtype=np.float32).reshape(-1, 6)


        # convert to range image to calculate offset distance on range image space
        img_gen, _ = pcd2range(scan_gen[:, :3], lossless=True, labels=scan_gen[:, 3:], **self.pcd_to_range_config)
        img_real, color_real = pcd2range(scan_real[:, :3], lossless=True, labels=scan_real[:, 3:], **self.pcd_to_range_config)

        # calculate offset distance
        offset_label = img_real - img_gen

        # convert back to point cloud
        pcd_real, _, _, mask_real = range2pcd_w_mask(img_real, depth_scale=1., log_scale=False, **self.pcd_to_range_config)
        pcd_gen, _, offset_label, mask_intersection = range2pcd_w_mask(img_gen, label=offset_label, input_mask=mask_real, depth_scale=1., log_scale=False, **self.pcd_to_range_config)
        pcd_real, _, _, _ = range2pcd_w_mask(img_real, input_mask=mask_intersection, depth_scale=1., log_scale=False, **self.pcd_to_range_config)

        color_real = color_real.reshape(-1, 3)[mask_intersection]
        # np.savez("data.npz", scan_gen=scan_gen, scan_real=scan_real, img_gen=img_gen, img_real=img_real, pcd_gen=pcd_gen, pcd_real=pcd_real, label=offset_label, idx=idx)
        # raise ValueError

        data_dict = dict(coord=pcd_gen, label=offset_label, coord_real=pcd_real, color_real=color_real, idx=idx)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{frame_name}"
        return data_name
