import torch.nn as nn
import torch
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from ..builder import MODELS, build_model


@MODELS.register_module()
class ImageModelWrapper(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = build_model(backbone)

    def forward(self, input_dict):
        x = input_dict["input"]
        B = x.shape[0]
        timesteps = torch.zeros(B, device=x.device).long()
        pred_offsets = self.backbone(x, timesteps)
        out_dict = dict(is_image_model=True, pred_offsets=pred_offsets)
        return out_dict
