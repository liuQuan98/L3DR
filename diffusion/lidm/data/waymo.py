import glob
import os
import pickle
import numpy as np
import yaml
from PIL import Image
import xml.etree.ElementTree as ET

from lidm.data.base import DatasetBase
from .annotated_dataset import Annotated3DObjectsDataset
from .conditional_builder.utils import corners_3d_to_2d
from .helper_types import Annotation
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd

# TODO add annotation categories and semantic categories

CATEGORIES = ["ignore", "car", "truck", "bus", "other vehicle", "person who rides a motorcycle",
              "person who rides a bicycle", "pedestrian", "sign", "traffic light", "pole",
              "construction cone", "bicycle", "motorcycle", "building", "vegetation",
              "tree trunk", "curb", "road", "lane marker", "other ground", "horizontal surface that can not drive",
              "surface when pedestrians most likely to walk on",]


CATE2LABEL = {k: v for v, k in enumerate(CATEGORIES)}
LABEL2RGB = np.array([(0, 0, 0), (0, 0, 142), (119, 11, 32), (0, 0, 230), (0, 0, 70), (0, 0, 90), (220, 20, 60),
                      (255, 0, 0), (0, 0, 110), (128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140),
                      (70, 70, 70), (190, 153, 153), (107, 142, 35), (0, 80, 100), (230, 150, 140), (153, 153, 153),
                      (220, 220, 0)])


class WODBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'waymo'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1
        self.img_size = kwargs['dataset_config'].size
        self.align_to_kitti = kwargs['dataset_config'].align_to_kitti
        self.fov = kwargs['dataset_config'].fov

        res = np.load("data/config/wod_lidar_params.npz")
        # self.inclinations = res["beam_inclinations"]
        # self.img_size = res["img_size"]
        self.vehice_to_lidar_transform = np.linalg.inv(res["lidar_to_vehicle_calib"])

        assert self.align_to_kitti # only support KITTI-style coloring
        self.label_to_rgb = LABEL2RGB

    def load_lidar_sweep(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        # strength = np.tanh(scan[:, -1].reshape([-1, 1]))
        coord = coord @ self.vehice_to_lidar_transform[:3, :3].T + self.vehice_to_lidar_transform[:3, 3]
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
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map, _ = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)

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


class WODDataset(WODBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in ['segmentation']  # for segmentation input only
        self.label2rgb = LABEL2RGB

    def prepare_data(self):
        # read data paths from WOD
        if isinstance(self.split, str):
            self.split = [self.split]
        for split in self.split:
            self.data.extend(glob.glob(
                os.path.join(self.data_root, split, "*", "velodyne", "*.bin")
            ))

    def align_2_kitti(self, label):
        mapping = [0, 1, 4, 5, 5, 8, 
                   7, 6, 19, 18, 18, 
                   14, 2, 3, 13, 15,
                   16, 11, 9, 9, 12, 12,
                   11]
        return np.array(mapping)[label]

    def load_semantic_map(self, path, pcd):
        label_file = path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                labels = (np.fromfile(a, dtype=np.int32).reshape(-1, 2)[:, 1])
        else:
            labels = np.zeros(pcd.shape[0]).astype(np.int32)

        if self.align_to_kitti:
            labels = self.align_2_kitti(labels)

        proj_range, sem_map = pcd2range(pcd, self.img_size, self.fov, self.depth_range, labels=labels)
        # sem_map = np.expand_dims(sem_map, axis=0).astype(np.int64)
        sem_map = sem_map.astype(np.int64)
        if self.filtered_map_cats is not None:
            sem_map[np.isin(sem_map, self.filtered_map_cats)] = 0  # set filtered category as noise
        onehot = np.eye(self.num_sem_cats, dtype=np.float32)[sem_map].transpose(2, 0, 1)
        return proj_range, onehot, sem_map


class WODTrain(WODDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/Waymo-Pointcept-format', split='training', **kwargs)


class WODValidation(WODDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/Waymo-Pointcept-format', split='validation', **kwargs)
