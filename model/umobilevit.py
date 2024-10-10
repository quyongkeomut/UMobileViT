from typing import Tuple, Callable
from math import prod
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    ReLU,
    Conv2d,
    GroupNorm,
    Upsample,
    Identity
)
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

from torch.nn.init import zeros_

from model.transfomer import (
    TransformerEncoderLayer, 
    TransformerDecoderLayer,
    _get_initializer
)
from model.encoder import UMobileViTEncoder
from model.decoder import UMobileViTDecoder
from model.seg_head import SegmentationHead



     
class UMobileViT(Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        input_size: int | Tuple[int, int] = (360, 640),
        d_model: int = 96,
        expansion_factor: float = 5,
        alpha: float = 1,
        patch_size: int | Tuple[int, int] = (3, 2),
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        num_transformer_block: int = 2,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        r"""
        Initializer of UMobileViT model

        Args:
            in_channels (int, optional): Number of channels of input image. Defaults to 3.
            alpha (float, optional): Controls the width of the model. Defaults to 1.
            patch_size (int | Tuple[int, int], optional): Size of patch using in encoder layer. 
                Defaults to 2.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
            norm_num_groups (int, optional): Control number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm. Defaults to 4.
            bias (bool, optional): If ``True``, add trainable bias to building blocks. Defaults to True.
            initializer (str | Callable[[Tensor], Tensor], optional): Parameters initializer.
                Defaults to "he_uniform" aka _kaiming_uniform.
        """
        assert (
            alpha >= 0 
        ), f"alpha must be greater than zero, got alpha={alpha}"

        super().__init__()    
        self.alpha = alpha
        d_model = int(alpha*d_model)
            
        kwargs = {
            "d_model": d_model,
            "expansion_factor": expansion_factor,
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias, 
            "num_transformer_block": num_transformer_block,
            "initializer": initializer,
            "device": device,
            "dtype": dtype
        }
        self.encoder = UMobileViTEncoder(in_channels=in_channels, **kwargs)
        self.decoder = UMobileViTDecoder(**kwargs)
        self.seg_head = SegmentationHead(
            in_channels=d_model,
            out_channels=out_channels,
            input_size= input_size 

        )
    
    def forward(self, input: Tensor) -> Tensor:
        encoder_outputs = self.encoder(input)
        output_decoder = self.decoder(tuple(reversed(encoder_outputs)))

        output = self.seg_head(output_decoder)
        return output