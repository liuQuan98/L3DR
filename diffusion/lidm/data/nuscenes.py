import glob
import os
import pickle
import numpy as np
import yaml
from PIL import Image
import xml.etree.ElementTree as ET

from typing import Sequence
from lidm.data.base import DatasetBase
from .annotated_dataset import Annotated3DObjectsDataset
from .conditional_builder.utils import corners_3d_to_2d
from .helper_types import Annotation
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd

# nuscenes
CATEGORIES = ["ignore", "barrier", "bicycle", "bus", "car", "construction vehicle",
              "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
              "path suitable or safe for driving", "other flat", "sidewalk", 
              "terrain", "man made", "vegetation",]
CATE2LABEL = {k: v for v, k in enumerate(CATEGORIES)}
LABEL2RGB = np.array([(0, 0, 0), (190, 153, 153), (100, 11, 32), (0, 120, 180), (0, 0, 142), (50, 50, 200), (0, 0, 230),
                      (220, 20, 60), (255, 255, 0), (0, 60, 255), (0, 0, 70), (128, 64, 128), (230, 150, 140),
                      (244, 35, 232), (230, 150, 140), (70, 70, 70), (107, 142, 35)])


class NUSCENESBase(DatasetBase):
    def __init__(self, **kwargs):
        self.sweeps= 1 
        super().__init__(**kwargs)
        self.dataset_name = 'nuscenes'
        self.ignore_index = -1 # nuscenes ignore is -1, valid categories start from 0.
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1
        self.img_size = kwargs['dataset_config'].size
        self.fov = kwargs['dataset_config'].fov

        # due to low but very wide images tend to change training dynamics, we double the number of lasers to 64
        # so that nuscenes shares the same training dynamics as semantickitti.
        # Our point cloud --> range image conversion is not based on the number of lasers.
        # Specifically, we match a uniform range image grid to the point cloud and round the points onto the grid. 
        # The final range image will have whatever shape we want, but might have horizontal hollow stripes, due to points not sufficient to fill in all pixels.
        # if this option is enabled, both range image and segmentation will be vertically copied and offset by 1 pixel
        # to fill in the horizontal gap.
        self.double_num_lasers_to_64 = kwargs['dataset_config'].double_num_lasers_to_64
        self.double_seg_only = kwargs['dataset_config'].double_seg_only

        self.label_to_rgb = LABEL2RGB

    def load_lidar_sweep(self, path):
        points = np.fromfile(str(path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        return coord

    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError
    
    def sem2rgb(self, sem_map):
        rgb_HWC = self.label_to_rgb[sem_map] # shape (H, W, C)
        return np.transpose(rgb_HWC, [2, 0, 1]) # now with shape (C, H, W)
    
    def process_rgb(self, rgb):
        rgb = rgb / 255.
        rgb = rgb * 2. - 1.
        rgb = np.clip(rgb, -1, 1)
        return rgb

    def __getitem__(self, idx):
        example = dict()

        data = self.data[idx]
        data_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map, _ = self.load_semantic_map(data, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)

        if self.double_num_lasers_to_64:
            # all images are in shape [C, H, W], c=1 for depth amd kask, c=num_sem_cats+1 for seg
            if not self.double_seg_only:
                assert proj_range.shape[0] == 1 and proj_mask.shape[0] == 1
                # move proj_range down by a pixel and add to original image
                proj_range_offset = np.concatenate([np.zeros_like(proj_range[:, :1, :]), proj_range[:, :-1, :]], axis=1)
                proj_mask_offset = np.concatenate([np.zeros_like(proj_mask[:, :1, :]), proj_mask[:, :-1, :]], axis=1)
                # merge all images
                proj_range = np.max([proj_range, proj_range_offset], axis=0)
                proj_mask = np.max([proj_mask, proj_mask_offset], axis=0)
            sem_map = example[self.condition_key]
            sem_map_offset = np.concatenate([np.zeros_like(sem_map[:, :1, :]), sem_map[:, :-1, :]], axis=1)
            sem_map_vacancies = np.where(sem_map[0, :, :] == 1)
            sem_map[:, sem_map_vacancies[0], sem_map_vacancies[1]] = sem_map_offset[:, sem_map_vacancies[0], sem_map_vacancies[1]]
            example[self.condition_key] = sem_map

        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras

        return example


class NUSCENESDataset(NUSCENESBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in ['segmentation']  # for segmentation input only
        self.label2rgb = LABEL2RGB
        self.learning_map = self.get_learning_map(self.ignore_index)

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def prepare_data(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        self.data = data_list

    def load_semantic_map(self, data, pcd):
        gt_segment_path = os.path.join(
            self.data_root, "raw", data["gt_segment_path"]
        )
        segment = np.fromfile(
            str(gt_segment_path), dtype=np.uint8, count=-1
        ).reshape([-1])
        labels = np.vectorize(self.learning_map.__getitem__)(segment).astype(
            np.int64
        )

        assert self.ignore_index == -1
        labels += 1 # convert from -1 to 0 based index (starting from ignore_index)

        proj_range, sem_map = pcd2range(pcd, self.img_size, self.fov, self.depth_range, labels=labels)
        # sem_map = np.expand_dims(sem_map, axis=0).astype(np.int64)
        sem_map = sem_map.astype(np.int64)
        onehot = np.eye(self.num_sem_cats, dtype=np.float32)[sem_map].transpose(2, 0, 1)
        return proj_range, onehot, sem_map


class NUSCENESTrain(NUSCENESDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/nuScenes-converted', split='train', **kwargs)


class NUSCENESValidation(NUSCENESDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/nuScenes-converted', split='val', **kwargs)

class NUSCENESAll(NUSCENESDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/nuScenes-converted', split=['train', 'val'], **kwargs)
