from typing import Tuple, Callable

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    ReLU,
    Conv2d,
    GroupNorm,
    Upsample,
)
from torch.nn.modules.utils import _pair

from torch.nn.init import zeros_

from model.module import (
    _get_initializer,
    _get_clones
)


def _get_upsample_block(
    in_channels: int,
    bias: bool,
    norm_num_groups: int,
    initializer: Callable,
    **kwargs,
) -> Sequential:
    # depthwise conv, followed by non-linearity and group norm
    block = Sequential(
        Upsample(
            scale_factor=kwargs["scale_factor"], 
            mode=kwargs["mode"]
        ),
        Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1),
            groups=in_channels,
            bias=bias,
            
        ),
        ReLU(),
        GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            device=kwargs["device"],
            dtype=kwargs["dtype"]
        ),
    )
    
    initializer(block[1].weight)
    if block[1].bias is not None:
        zeros_(block[1].bias)
    
    return block
    

class SegmentationHead(Module):
    def __init__(self, 
        d_model: int = 96,
        alpha: float = 1,
        out_channels: int | Tuple[int, int] = [2, 2],
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        **kwargs
    ) -> None:
        
        super().__init__()
        
        upsampling_kwargs = {"scale_factor": 2, "mode": "nearest"}
        factory_kwargs = {"device": device, "dtype": dtype}
        initializer = _get_initializer(initializer)
        self.initializer = initializer
        in_channels: int = int(alpha*d_model)
        out_channels = _pair(out_channels)
        new_space_dim = in_channels
        
        # init the upsample part, which share the same structure at each head
        upsample_head = Sequential(
            # project to new space
            Conv2d(
                in_channels=in_channels,
                out_channels=new_space_dim,
                kernel_size=1,
                bias=bias,
                **factory_kwargs
            ),
            ReLU(),
            GroupNorm(
                num_groups=norm_num_groups,
                num_channels=new_space_dim,
                **factory_kwargs
            ),
        )
        # upsize to 8 times 
        upsample_head.extend(
            _get_clones(
                _get_upsample_block(
                    in_channels=new_space_dim,
                    bias=bias,
                    norm_num_groups=norm_num_groups,
                    initializer=self.initializer,
                    **upsampling_kwargs, 
                    **factory_kwargs
                ) 
            , 3)
        )
        
        self.lane_head, self.drivable_head = _get_clones(upsample_head, 2)
        
        # lane head
        self.lane_head.extend([
            # output conv
            Conv2d(
                in_channels=new_space_dim,
                out_channels=new_space_dim,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                groups=new_space_dim,
                bias=bias,
            ),
            ReLU(),
            Conv2d(
                in_channels=new_space_dim,
                out_channels=out_channels[0],
                kernel_size=1,
                bias=bias,
                **factory_kwargs
            ),
        ])
        
        # drivable head 
        self.drivable_head.extend([
            # output conv
            Conv2d(
                in_channels=new_space_dim,
                out_channels=new_space_dim,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                groups=new_space_dim,
                bias=bias,
            ),
            ReLU(),
            Conv2d(
                in_channels=new_space_dim,
                out_channels=out_channels[1],
                kernel_size=1,
                bias=bias,
                **factory_kwargs
            ),  
        ])

        self._reset_parameters()
    
    
    def _reset_parameters(self,) -> None:
        for layer in self.lane_head:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)

        for layer in self.drivable_head:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
                

    def forward(self, inputs: Tensor) -> Tuple[Tensor]:
        line_mask = self.lane_head.forward(inputs)
        drivable_mask = self.drivable_head.forward(inputs)
        return (drivable_mask, line_mask)


if __name__ == "__main__":
    
    from torchinfo import summary
    from fvcore import nn as fnn

    model = SegmentationHead(
        in_channels= 48,
        out_channels= 2
    )

    input = torch.randn(1, 48, 72, 128)

    model_inputs = (input, )
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops, max_depth=5))
    summary(model, input_data=input, depth=5)