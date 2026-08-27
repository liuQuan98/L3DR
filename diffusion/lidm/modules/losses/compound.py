import torch
from torch import nn

from . import weights_init, l1, l2, hinge_d_loss, vanilla_d_loss, measure_perplexity, square_dist_loss
from .geometric import GeoConverter
from .discriminator import NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2
from .perceptual import PerceptualLoss
from ...utils.misc_utils import instantiate_from_config


class DepthSemanticCompoundLoss(nn.Module):
    ''' Compounded semseg and depth loss. Defaults to first channel as depth and three following channels as mapped RGB semantic map. '''
    def __init__(self, config_depth, config_sem, dataset_config=dict()):
        super().__init__()
        config_depth.params.dataset_config = dataset_config
        config_sem.params.dataset_config = dataset_config
        self.depth_loss = instantiate_from_config(config_depth)
        self.sem_loss = instantiate_from_config(config_sem)

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None):
        loss_depth, log_depth = self.depth_loss(codebook_loss, inputs[:, :1, :, :], reconstructions[:, :1, :, :], optimizer_idx,
                                                global_step, last_layer, cond, split, predicted_indices, masks)
        loss_sem,   log_sem   =   self.sem_loss(codebook_loss, inputs[:, 1:4, :, :], reconstructions[:, 1:4, :, :], optimizer_idx,
                                                global_step, last_layer, cond, split, predicted_indices, masks)
        log = {}
        for k, v in log_depth.items():
            log["depth_"+k] = v
        for k, v in log_sem.items():
            log["sem_"+k] = v
        if optimizer_idx == 0:
            log[split + "/rec_loss"] = log_depth[split + "/rec_loss"] + log_sem[split + "/rec_loss"]
        return loss_depth + loss_sem, log
