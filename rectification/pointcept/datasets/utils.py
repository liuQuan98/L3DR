"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
import open3d as o3d
from torch.utils.data.dataloader import default_collate


def make_open3d_point_cloud(xyz, color=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd


def match_contrastive_group(all_coord, offset, list_trans, K, search_voxel_size):
    '''
    Finds KNN for all points of center_pc w.r.t. all point clouds inside neighbourhood_pc.
    We assume all input point clouds and xyzs have their LiDAR centers located exactly at the origin.
    Return value is a three-tuple:
        group:        The length for all co-location groups, in series.
        index:        All indexes of matched local patch. You should run 'torch.split(index, group)' to split this into local groups.
        finest_flag:  Has same length as 'index', with boolean values indicating whether a point is a finest neighbor.
    '''
    all_pc_copy = all_coord.cpu().detach().numpy()

    center_pc_copy = all_pc_copy[0:offset[0], :]

    center_KDTree = o3d.geometry.KDTreeFlann(make_open3d_point_cloud(center_pc_copy))
    neighborhood_xyz = []
    neighborhood_KDTrees = []
    for i in range(len(offset)-1):
        neighborhood_xyz.append(all_pc_copy[offset[i]:offset[i+1]])
        neighborhood_KDTrees.append(o3d.geometry.KDTreeFlann(make_open3d_point_cloud(neighborhood_xyz[i])))

    group = []
    index = []
    finest_flag = []
    central_xyz = center_pc_copy
    for i, point in enumerate(central_xyz):
        # Closest dist is set to its own center distance by default.
        # If no other finer points are found, then the central representation acts natually as the finest neighbor
        closest_center_dist = np.linalg.norm(point)
        [_, knn_nghb_indices, _] = center_KDTree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            knn_nghb_indices = knn_nghb_indices[:K]
        knn_nghb_indices = list(knn_nghb_indices)
        original_knn_length = len(knn_nghb_indices)
        # If no finer point is found, then 'finest_position' is 0, 
        # which is exactly the position of central point i in 'index'.
        finest_position = 0
        current_start_idx = len(central_xyz)
        for j, KDTree in enumerate(neighborhood_KDTrees):
            central_coordinate = list_trans[j][0, :3, 3].reshape(3).detach().cpu().numpy()
            [_, idx, _] = KDTree.search_radius_vector_3d(point, search_voxel_size)
            if K is not None:
                idx = idx[:K]
            if idx:
                dist = np.linalg.norm(neighborhood_xyz[j][idx[0]] - central_coordinate)
                if dist < closest_center_dist:
                    closest_center_dist = dist
                    finest_position = len(knn_nghb_indices)
                knn_nghb_indices += list(np.array(idx) + current_start_idx) # idx is a list of indices
            current_start_idx += len(neighborhood_xyz[j])
        if len(knn_nghb_indices) == original_knn_length: # no matches are found in any of the neighborhood point clouds, then skip
            continue

        group.append(len(knn_nghb_indices))

        index += knn_nghb_indices
        local_finest_flag = list(np.zeros(len(knn_nghb_indices)))
        local_finest_flag[finest_position] = 1
        finest_flag += local_finest_flag

    return (
        torch.tensor(group, dtype=torch.long), 
        torch.tensor(index, dtype=torch.long), 
        torch.tensor(finest_flag, dtype=torch.bool),
    )


def _exhaustive_hash(index_split, M):
    result = []
    if len(index_split)==0:
        return [] 
    for indexes in index_split:
        indexes = np.array(indexes.cpu())
        for i in range(len(indexes)-1):
            result.append(np.min([indexes[i] + indexes[i+1 : len(indexes)] * M, 
                                  indexes[i] * M + indexes[i+1 : len(indexes)]], axis=0, keepdims=False))
    return np.concatenate(result, axis=0)


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch_tmp = {}
        GCL_flag = False
        # special inter-item and intra-item collating (i.e., 2d flattening) for GCL
        if "gcl_all_pc_dicts" in batch[0]:
            GCL_flag = True
            matrix = [item["gcl_all_pc_dicts"] for item in batch] # matrix: B*(\phi+1), matrix[0][0] is a dict for a single point cloud
            h, w = len(matrix), len(matrix[0])
            for key in matrix[0][0]:
                batch_tmp[key] = collate_fn([matrix[index//w][index%w][key] for index in range(int(h*w))])
                if key == "coord":
                    batch_tmp["offset"] = torch.tensor([len(matrix[index//w][index%w][key]) for index in range(int(h*w))]).long()

        batch_tmp.update({key: collate_fn([d[key] for d in batch]) 
                          for key in batch[0] if key != "gcl_all_pc_dicts"})
        
        batch = batch_tmp # overwrite original dict
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)

        if GCL_flag:
            # we currently only support batch size 1
            group, index, finest_flag = match_contrastive_group(
                batch["origin_coord"],
                batch["offset"],
                batch["list_M"],
                batch["matching_max_k"],
                batch["matching_max_radius"],
            )
            # print(f"found {len(group)} groups of correspondence in dataset {batch['condition'][0]}.")
            index_hash = _exhaustive_hash(torch.split(index, tuple(group.tolist())), len(batch["origin_coord"]))
            # if index_hash == []:
            #     print("Empty index hash! Printing batch.")
            #     print(batch["origin_coord"],)
            #     print(batch["offset"],)
            #     print(batch["list_M"],)
            #     print(batch["matching_max_k"],)
            #     print(batch["matching_max_radius"],)
            #     print(batch["condition"])
            #     raise ValueError
            batch["group"] = group
            batch["index"] = index
            batch["finest_flag"] = finest_flag
            batch["index_hash"] = index_hash
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
