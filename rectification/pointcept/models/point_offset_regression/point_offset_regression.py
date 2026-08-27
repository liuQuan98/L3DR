import torch.nn as nn
import torch
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from ..builder import MODELS, build_model


@MODELS.register_module()
class PointOffsetRegressor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        # if "condition" in input_dict.keys():
        #     # PPT (https://arxiv.org/abs/2308.09718)
        #     # currently, only support one batch one condition
        #     input_dict["condition"] = input_dict["condition"][0]
        pred_offsets = self.backbone(input_dict)
        if type(pred_offsets) is dict and pred_offsets["is_image_model"]:
            pred_offsets = pred_offsets["pred_offsets"]
            unified_output = pred_offsets
        else:
            if type(pred_offsets) is Point:
                pred_offsets = pred_offsets.feat
            if pred_offsets.shape[-1] >= 3:
                offset_vectors = pred_offsets[..., :3]
                remaining_output = pred_offsets[..., 3:]
                if input_dict.get("origin_coord", None) is not None:
                    # use the original, unaugmented point cloud as target to provide correct supervision. 
                    # This is activated when data augmentation is applied.
                    target_pcd = input_dict["origin_coord"] 
                else:
                    target_pcd = input_dict["coord"]
                # normalize direction vectors, and use the projection of output vector on radial direction as the final offset
                directions = target_pcd / (torch.norm(target_pcd, dim=-1, keepdim=True) + 1e-6)
                offset_scalars = (offset_vectors * directions).sum(dim=-1, keepdim=True)
                if pred_offsets.shape[-1] >= 4:
                    unified_output = torch.cat([offset_scalars, remaining_output], dim=-1)
                    assert len(pred_offsets.shape) == 2
                    pred_offsets = offset_scalars.flatten()
                else:
                    unified_output = offset_scalars.flatten()
                    pred_offsets = unified_output
        # train
        if self.training:
            loss = self.criteria(unified_output, input_dict["label"])
            return dict(loss=loss)
        # eval
        elif "label" in input_dict.keys():
            loss = self.criteria(unified_output, input_dict["label"])
            return dict(loss=loss, pred_result=pred_offsets)
        # test
        else:
            if "inverse" not in input_dict.keys():
                input_dict["inverse"] = None
            return dict(pred_result=pred_offsets, inverse=input_dict["inverse"])
