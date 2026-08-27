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

from .builder import DATASETS
from .defaults import DefaultDataset
from .lidar_utils import range2pcd, range2pcd_w_mask, pcd2range


@DATASETS.register_module()
class SemanticKITTIGen2RealImageDataset(DefaultDataset):    
    def __init__(
        self,
        split="train",
        data_root="data/lidar-diffusion-official-semseg-log",
        pred_root=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        with_sem_input=False,
        loop=1,
        pcd_to_range_config=None,
    ):
        self.pred_root = pred_root
        self.pcd_to_range_config = pcd_to_range_config
        self.with_sem_input = with_sem_input

        # ensure the transform process do not calclate offsets
        # which completely rules out its compatibility with the point models
        for i in range(len(transform)):
            if transform[i]["type"] == "Collect":
                transform[i]["offset_keys_dict"] = {}
                break

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
    
    def process_scan(self, range_img, log_scale=False, depth_scale=None, **kwargs):
        range_img = np.where(range_img < 0, 0, range_img)

        if log_scale:
            # log scale
            range_img = np.log2(range_img + 0.0001 + 1)

        range_img = range_img / depth_scale
        range_img = range_img * 2. - 1.

        range_img = np.clip(range_img, -1, 1)
        range_img = np.expand_dims(range_img, axis=0)

        # mask
        if log_scale:
            depth_thresh = (np.log2(1./255. + 1) / depth_scale) * 2. - 1 + 1e-6
        else:
            depth_thresh = (1./255. / depth_scale) * 2. - 1 + 1e-6
        range_mask = np.ones_like(range_img)
        range_mask[range_img < depth_thresh] = -1

        return range_img, range_mask
    
    def rescale_rgb(self, rgb):
        rgb = rgb * 2. - 1.
        rgb = np.clip(rgb, -1, 1)
        return rgb
        
    def get_data(self, idx):
        # image file names
        gen_data_path = self.data_list[idx % len(self.data_list)]

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
        # notice that color loaded from this source is already in [0, 1] range, but depth is in [0, 56]
        # so we need to convert depth to [-1, 1] range with many configs, but only need to rescale color to be within [-1, 1] range
        img_gen, _ = pcd2range(scan_gen[:, :3], lossless=True, labels=scan_gen[:, 3:], **self.pcd_to_range_config)
        img_real, color_real = pcd2range(scan_real[:, :3], lossless=True, labels=scan_real[:, 3:], **self.pcd_to_range_config)

        # calculate offset distance
        H, W = self.pcd_to_range_config["size"]
        offset_label = img_real - img_gen # [H, W]

        img_gen,  _ = self.process_scan(img_gen, **self.pcd_to_range_config) # [1, H, W]
        img_real, _ = self.process_scan(img_real, **self.pcd_to_range_config) # [1, H, W]
        color_real  = self.rescale_rgb(color_real) # [H, W, 3]

        color_real  = color_real.transpose(2, 0, 1) # [3, H, W]
        offset_label= offset_label.reshape(1, H, W)

        if self.with_sem_input:
            img_gen = np.concatenate([img_gen, img_gen], axis=0)

        data_dict = dict(input=img_gen, label=offset_label, target=img_real, color_real=color_real, idx=idx)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{frame_name}"
        return data_name
