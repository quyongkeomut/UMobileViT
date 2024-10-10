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

from torchvision.transforms import Resize



class SegmentationHead(Module):
    def __init__(self, 
        in_channels: int = 3,
        out_channels: int = 3,
        input_size: int | Tuple[int, int] = (360, 640),
        d_model: int = 92,
        expansion_factor: float = 4,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        num_transformer_block: int = 2,
        device=None,
        dtype=None,
        **kwargs
        
        ) -> None:
        super().__init__()

        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        self.lane_head = Sequential(
            Upsample(**upsampling_kwargs),
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels//2,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype),
            ReLU(),
            GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels//2,
                device=device,
                dtype=dtype),
            Upsample(**upsampling_kwargs),
            Conv2d(
                in_channels=in_channels//2,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype),
            ReLU(),
            GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                device=device,
                dtype=dtype),

        )

        self.drivable_head = Sequential(
            Upsample(**upsampling_kwargs),
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels//2,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype),
            ReLU(),
            GroupNorm(
                num_groups=norm_num_groups//2,
                num_channels=in_channels//2,
                device=device,
                dtype=dtype),
            Upsample(**upsampling_kwargs),
            Conv2d(
                in_channels=in_channels//2,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype),
            ReLU(),
            GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                device=device,
                dtype=dtype),
        )

        self.resize = Resize(input_size)

    def forward(self, inputs):

        line_mask = self.lane_head.forward(inputs)
        line_mask = self.resize.forward(line_mask)

        drivable_mask = self.drivable_head.forward(inputs)
        drivable_mask = self.resize.forward(drivable_mask)

        return (drivable_mask, line_mask)

if __name__ == "__main__":
    
    from torchinfo import summary
    from fvcore import nn as fnn

    model = SegmentationHead(
        in_channels= 48,
        out_channels= 3
    )

    input = torch.randn(1, 48, 72, 128)

    model_inputs = (input, )
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops, max_depth=5))
    summary(model, input_data=input, depth=5)