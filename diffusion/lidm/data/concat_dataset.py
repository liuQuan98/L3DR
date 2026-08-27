import os
import numpy as np
from torch.utils.data import Dataset

from lidm.utils.misc_utils import instantiate_from_config


class ConcatDataset(Dataset):
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [instantiate_from_config(dataset) for dataset in datasets]
        # let the longest dataset be the first, which accomodates to specifications of MultiDatasetDataloader
        self.datasets = sorted(self.datasets, key=lambda cls:cls.__len__(), reverse=True)
        self.loop = loop
        self.data = self.get_data_list()
        self.idx_perm = np.random.permutation(len(self.data_list))

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i])) * i, np.arange(len(self.datasets[i]))
                )
            )
        return data_list

    def get_data(self, idx):
        mapped_idx = self.idx_perm[idx % len(self.data_list)]
        # self.logger.info(f"idx: {idx}, mapped: {mapped_idx}")
        dataset_idx, data_idx = self.data_list[mapped_idx]
        return self.datasets[int(dataset_idx)][int(data_idx)]

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop