from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg

from ..replace_hf import replace_linear_with_bitlinearb158


def bit_deit_tiny_patch16_224_b158(num_classes=1000):
    model = VisionTransformer(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model.default_cfg = _cfg()
    replace_linear_with_bitlinearb158(model)
    return model
