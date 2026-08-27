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
CATEGORIES = ['ignore', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
              'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
              'pole', 'traffic-sign']
CATE2LABEL = {k: v for v, k in enumerate(CATEGORIES)}  # 0: invalid, 1~10: categories
LABEL2RGB = np.array([(0, 0, 0), (0, 0, 142), (119, 11, 32), (0, 0, 230), (0, 0, 70), (0, 0, 90), (220, 20, 60),
                      (255, 0, 0), (0, 0, 110), (128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140),
                      (70, 70, 70), (190, 153, 153), (107, 142, 35), (0, 80, 100), (230, 150, 140), (153, 153, 153),
                      (220, 220, 0)])
CAMERAS = ['CAM_FRONT']
BBOX_CATS = ['car', 'people', 'cycle']
BBOX_CAT2LABEL = {'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'person': 1, 'rider': 2, 'motorcycle': 2, 'bicycle': 2}

# train + test
SEM_KITTI_TRAIN_SET = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
KITTI_TRAIN_SET = SEM_KITTI_TRAIN_SET + ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
KITTI360_TRAIN_SET = ['00', '02', '04', '05', '06', '07', '09', '10'] + ['08']  # partial test data at '02' sequence
CAM_KITTI360_TRAIN_SET = ['00', '04', '05', '06', '07', '08', '09', '10']  # cam mismatch lidar in '02'

# validation
SEM_KITTI_VAL_SET = KITTI_VAL_SET = ['08']
CAM_KITTI360_VAL_SET = KITTI360_VAL_SET = ['03']


class KITTIBaseWithSem(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'kitti'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1
        self.img_size = kwargs['dataset_config'].size
        self.label_to_rgb = LABEL2RGB

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

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

        # semantic maps
        proj_range, sem_onehot, sem_indice = self.load_semantic_map(data_path, sweep)
        example[self.condition_key] = sem_onehot
        proj_range, proj_mask = self.process_scan(proj_range)

        example['mask'] = proj_mask

        sem_as_rgb = self.sem2rgb(sem_indice)
        sem_as_rgb = self.process_rgb(sem_as_rgb)
        example['image'] = np.zeros((4, *self.img_size), dtype=np.float32)
        example['image'][0, :, :] = proj_range
        example['image'][1:4, :, :] = sem_as_rgb
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


class SemanticKITTIBaseWithSemInput(KITTIBaseWithSem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in ["None"]  # for autoencoder training of unconditional simultaneous generation of LiDAR and semantic map
        self.label2rgb = LABEL2RGB

    def prepare_data(self):
        # read data paths from KITTI
        for seq_id in eval('SEM_KITTI_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'dataset/sequences/{seq_id}/velodyne/*.bin')))
        # read label mapping
        data_config = yaml.safe_load(open('./data/config/semantic-kitti.yaml', 'r'))
        remap_dict = data_config["learning_map"]
        max_key = max(remap_dict.keys())
        self.learning_map = np.zeros((max_key + 100), dtype=np.int32)
        self.learning_map[list(remap_dict.keys())] = list(remap_dict.values())

    def load_semantic_map(self, path, pcd):
        label_path = path.replace('velodyne', 'labels').replace('.bin', '.label')
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # semantic label in lower half
        labels = self.learning_map[labels]

        proj_range, sem_map = pcd2range(pcd, self.img_size, self.fov, self.depth_range, labels=labels)
        # sem_map = np.expand_dims(sem_map, axis=0).astype(np.int64)
        sem_map = sem_map.astype(np.int64)
        if self.filtered_map_cats is not None:
            sem_map[np.isin(sem_map, self.filtered_map_cats)] = 0  # set filtered category as noise
        onehot = np.eye(self.num_sem_cats, dtype=np.float32)[sem_map].transpose(2, 0, 1)
        return proj_range, onehot, sem_map


class SemanticKITTITrainWithSemInput(SemanticKITTIBaseWithSemInput):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/SemanticKITTI', split='train', **kwargs)


class SemanticKITTIValidationWithSemInput(SemanticKITTIBaseWithSemInput):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/SemanticKITTI', split='val', **kwargs)

