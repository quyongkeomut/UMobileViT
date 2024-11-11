from typing import Tuple, Callable

import torch
from torch import Tensor, cat
from torch.nn import (
    Module,
    Sequential,
    ModuleList,
    ReLU,
    Conv2d,
)
from torch.nn.modules.utils import _pair

from torch.nn.init import zeros_

from model.module import (
    _get_initializer,
)
from model.decoder import (
    UMobileViTDecoderAdditiveLayer,
    _get_upsample_block
)


class UpsampleHead(Module):
    def __init__(
        self,  
        d_model: int = 64,
        expansion_factor: float = 3,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        r"""
        Upsample Head is the module that restore the resolution similar to the original input.

        Args:
            d_model (int, optional): Dimentions / Number of feature maps of the whole model. 
                Defaults to 64.
            expansion_factor (float, optional): Expansion factor in Inverted Residual block. 
                Defaults to 3.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
            norm_num_groups (int, optional): Control number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm. Defaults to 4.
            bias (bool, optional): If ``True``, add trainable bias to building blocks. Defaults to True.
            initializer (str | Callable[[Tensor], Tensor], optional): Parameters initializer.
                Defaults to "he_uniform" aka _kaiming_uniform.
        """
        super().__init__()
        self.initializer = _get_initializer(initializer)
        
        factory_kwargs = {"device": device, "dtype": dtype}
        decoder_layer_kwargs = {
            "in_channels": d_model,
            "expansion_factor": expansion_factor,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "initializer": initializer,
        }
        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        upsampling_conv_kwargs = {
            "in_channels": d_model,
            "out_channels": d_model,
            "kernel_size": 3,
            "stride": 1,
            "padding": (1, 1),
            "groups": d_model,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
        }
        
        upsample_head_x2 = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
            UMobileViTDecoderAdditiveLayer(
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,
            )
        ])
        
        upsample_head_x4 = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
            UMobileViTDecoderAdditiveLayer(
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,)
        ])
        
        upsample_head_x8 = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
        ])
        
        self.layers = ModuleList([
            upsample_head_x2,
            upsample_head_x4, 
            upsample_head_x8,
        ])
        
    
    def forward(
        self,
        input: Tensor,
        outputs_stem: Tuple[Tensor]
    ) -> Tensor:
        assert (
            len(outputs_stem) == len(self.layers) - 1
        ), f"length of output_stems must be {len(self.layers) - 1}, got {len(outputs_stem)}"
        
        Z = input
        for i, output_stem in enumerate(outputs_stem):
            Z = self.layers[i][0](Z) # upsample
            for layer in self.layers[i][1:]: # additive layer
                Z = layer(Z, output_stem)
        Z = self.layers[-1][0](Z) # just upsample
        
        return Z
        

class SegmentationHead(Module):
    def __init__(
        self, 
        d_model: int,
        expansion_factor: float = 3,
        dropout_p: float = 0.1,
        out_channels: int | Tuple[int, int] = [2, 2],
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        r"""
        Upsample Head is the module that generate segmentation mask for Line and Drivable area
        Segmentation task.

        Args:
            d_model (int): Dimentions / Number of feature maps of the whole model. 
            expansion_factor (float, optional): Expansion factor in Inverted Residual block.
                Defaults to 3.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
            out_channels (int | Tuple[int, int], optional): Number of channels of output massk. 
                Defaults to (2, 2).
            norm_num_groups (int, optional): Control number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm. Defaults to 4.
            bias (bool, optional): If ``True``, add trainable bias to building blocks. Defaults to 
                True.
            initializer (str | Callable[[Tensor], Tensor], optional): Parameters initializer.
                Defaults to "he_uniform" aka _kaiming_uniform.
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        d_new = max(16, d_model//4)
        initializer = _get_initializer(initializer)
        self.initializer = initializer
        out_channels = _pair(out_channels)
        
        self.lane_head = ModuleList([
            Conv2d(
                in_channels=d_model,
                out_channels=d_new,
                kernel_size=1,
                bias=bias,
                **factory_kwargs    
            ),
            UpsampleHead(
                d_model=d_new,
                expansion_factor=expansion_factor,
                dropout_p=dropout_p,
                norm_num_groups=norm_num_groups,
                bias=bias,
                initializer=initializer,
                **kwargs,
                **factory_kwargs
            ),
            Sequential(
                Conv2d(
                    in_channels=d_new,
                    out_channels=d_new,
                    kernel_size=3,
                    stride=1,
                    padding=(1, 1),
                    groups=d_new,
                    bias=bias,
                    **factory_kwargs,
                ),
                ReLU(),
                Conv2d(
                    in_channels=d_new,
                    out_channels=out_channels[0],
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs
                ),
            )
        ])
        self.drivable_head = ModuleList([
            Conv2d(
                in_channels=d_model,
                out_channels=d_new,
                kernel_size=1,
                bias=bias,
                **factory_kwargs    
            ),
            UpsampleHead(
                d_model=d_new,
                expansion_factor=expansion_factor,
                dropout_p=dropout_p,
                norm_num_groups=norm_num_groups,
                bias=bias,
                initializer=initializer,
                **kwargs,
                **factory_kwargs
            ),
            Sequential(
                Conv2d(
                    in_channels=d_new,
                    out_channels=d_new,
                    kernel_size=3,
                    stride=1,
                    padding=(1, 1),
                    groups=d_new,
                    bias=bias,
                    **factory_kwargs,
                ),
                ReLU(),
                Conv2d(
                    in_channels=d_new,
                    out_channels=out_channels[1],
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs
                ),
            )
        ])

        self._reset_parameters()
    
    
    def _reset_parameters(self,) -> None:
        self.initializer(self.lane_head[0].weight)
        if self.lane_head[0].bias is not None:
            zeros_(self.lane_head[0].bias)
        
        self.initializer(self.drivable_head[0].weight)
        if self.drivable_head[0].bias is not None:
            zeros_(self.drivable_head[0].bias)
        
        for layer in self.lane_head[-1]:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)

        for layer in self.drivable_head[-1]:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
                

    def forward(
        self, 
        input: Tensor, 
        outputs_stem: Tensor
    ) -> Tuple[Tensor]:
        line_head_feat = self.lane_head[0](input)
        line_head_feat = self.lane_head[1](line_head_feat, outputs_stem)
        line_mask = self.lane_head[-1](line_head_feat)
        
        drivable_head_feat = self.drivable_head[0](input)
        drivable_head_feat = self.drivable_head[1](drivable_head_feat, outputs_stem)
        drivable_mask = self.drivable_head[-1](drivable_head_feat)
        
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