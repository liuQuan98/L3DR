from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier

# Backbones used by L3DR's Residual Regression Network
from .sparse_unet import *            # SpUNet-v1m1 (paper default backbone)
from .point_transformer_v3 import *   # PT-v3m1 (Table 3 ablation)
from .image_model import *            # 2D UNet control (Table 3, `data.is_image_model`)

# The RRN itself
from .point_offset_regression import *
