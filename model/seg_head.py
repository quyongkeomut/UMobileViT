from typing import Tuple, Callable

import torch
from torch import Tensor
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
    UMobileViTDecoderConcatLayer,
    _get_upsample_block
)


class UpsampleHead(Module):
    def __init__(
        self,  
        d_model: int = 64,
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
            d_model (int, optional): Dimensions / Number of feature maps of the whole model. 
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
            "kernel_size": 3,
            "stride": 1,
            "padding": (1, 1),
            "norm_num_groups": norm_num_groups,
            "bias": bias,
        }
        
        upsample_head_x2 = ModuleList([
           Sequential(
                Conv2d(
                    in_channels=d_model,
                    out_channels=d_model//2,
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    in_channels=d_model//2,
                    out_channels=d_model//2,
                    groups=d_model//2,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            ),
            UMobileViTDecoderConcatLayer(
                in_channels=d_model//2,
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,
            )
        ])
        
        upsample_head_x4 = ModuleList([
            Sequential(
                Conv2d(
                    in_channels=d_model//2,
                    out_channels=d_model//4,
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    in_channels= d_model//4,
                    out_channels=d_model//4,
                    groups=d_model//4,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            ),
            UMobileViTDecoderConcatLayer(
                in_channels=d_model//4,
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,)
        ])
        
        upsample_head_x8 = ModuleList([
            Sequential(
                Conv2d(
                    in_channels=d_model//4,
                    out_channels=d_model//8,
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    in_channels=d_model//8,
                    out_channels=d_model//8,
                    groups=d_model//8,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            )
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
 
 
class SegHead(Module):
    def __init__(
        self, 
        d_model: int,
        dropout_p: float = 0.1,
        out_channels: int = 2,
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        r"""
        SegHead is the module that generate segmentation mask for Segmentation task.

        Args:
            d_model (int): Dimensions / Number of feature maps of the whole model. 
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
        initializer = _get_initializer(initializer)
        self.initializer = initializer
        
        self.seg_head = ModuleList([
            UpsampleHead(
                d_model=d_model,
                dropout_p=dropout_p,
                norm_num_groups=norm_num_groups,
                bias=bias,
                initializer=initializer,
                **kwargs,
                **factory_kwargs
            ),
            Sequential(
                Conv2d(
                    in_channels=d_model//8,
                    out_channels=d_model//8,
                    kernel_size=3,
                    stride=1,
                    padding=(1, 1),
                    groups=d_model//8,
                    bias=bias,
                    **factory_kwargs,
                ),
                ReLU(),
                Conv2d(
                    in_channels=d_model//8,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs
                ),
            )
        ])

        self._reset_parameters()
    
    
    def _reset_parameters(self,) -> None:
        for layer in self.seg_head[-1]:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
                

    def forward(
        self, 
        input: Tensor, 
        outputs_stem: Tensor
    ) -> Tensor:
        r"""
        Forward method of Segmentation Head

        Args:
            input (Tensor): Ouput of decoder
            outputs_stem (Tensor): Outputs of stem blocks

        Returns:
            Tuple[Tensor]: Segmentation mask
        """
        seg_head_feat = self.seg_head[0](input, outputs_stem)
        seg_mask = self.seg_head[-1](seg_head_feat)

        return seg_mask
        
class DrivableAndLaneSegHead(Module):
    def __init__(
        self, 
        d_model: int,
        dropout_p: float = 0.1,
        out_channels: int | Tuple[int, int] = (2, 2),
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        r"""
        LineAndDrivableSegHead is the module that generate segmentation masks for Line and 
        Drivable area Segmentation task.

        Args:
            d_model (int): Dimensions / Number of feature maps of the whole model. 
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
        
        out_channels = _pair(out_channels)
        kwargs = {
            "d_model": d_model,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "initializer": initializer,
            "device": device,
            "dtype": dtype,
            **kwargs
        }
        
        # drivable head and lane head are segmentation heads
        self.lane_head = SegHead(out_channels=out_channels[0], **kwargs)
        self.drivable_head = SegHead(out_channels=out_channels[1], **kwargs)
                

    def forward(
        self, 
        input: Tensor, 
        outputs_stem: Tensor
    ) -> Tuple[Tensor]:
        r"""
        Forward method of Line and Drivable Segmentations

        Args:
            input (Tensor): Output of decoder
            outputs_stem (Tensor): Outputs of stem block 

        Returns:
            Tuple[Tensor]: Drivable mask and Line mask
        """
        drivable_mask = self.drivable_head(input, outputs_stem)
        lane_mask = self.lane_head(input, outputs_stem)
        
        return (drivable_mask, lane_mask)


# class BDD10KSegHead(Module):
#     def __init__(
#         self, 
#         d_model: int,
#         dropout_p: float = 0.1,
#         out_channels: int | Tuple[int, int] = (20),
#         norm_num_groups: int = 4,
#         bias: bool = True, 
#         initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
#         device=None,
#         dtype=None,
#         **kwargs,
#     ) -> None:
#         r"""
#         BDD10KSegHead is the module that generate segmentation masks for BDD100K Sementic Segmentation task.

#         Args:
#             d_model (int): Dimensions / Number of feature maps of the whole model. 
#             expansion_factor (float, optional): Expansion factor in Inverted Residual block.
#                 Defaults to 3.
#             dropout_p (float, optional): Dropout probability. Defaults to 0.1.
#             out_channels (int | Tuple[int], optional): Number of channels of output massk. 
#                 Defaults to (20).
#             norm_num_groups (int, optional): Control number of groups to be normalized. If
#                 ``norm_num_groups`` = 1, this is equivalent to LayerNorm. Defaults to 4.
#             bias (bool, optional): If ``True``, add trainable bias to building blocks. Defaults to 
#                 True.
#             initializer (str | Callable[[Tensor], Tensor], optional): Parameters initializer.
#                 Defaults to "he_uniform" aka _kaiming_uniform.
#         """
#         super().__init__()
        
#         # out_channels = _pair(out_channels)
#         kwargs = {
#             "d_model": d_model,
#             "dropout_p": dropout_p,
#             "norm_num_groups": norm_num_groups,
#             "bias": bias,
#             "initializer": initializer,
#             "device": device,
#             "dtype": dtype,
#             **kwargs
#         }
        
#         self.seg_head = SegHead(out_channels=out_channels[0], **kwargs)
                

#     def forward(
#         self, 
#         input: Tensor, 
#         outputs_stem: Tensor
#     ) -> Tensor:
#         r"""
#         Forward method of Line and Drivable Segmentations

#         Args:
#             input (Tensor): Output of decoder
#             outputs_stem (Tensor): Outputs of stem block 

#         Returns:
#             Tensor: Segmentation mask
#         """
        
#         seg_mask = self.seg_head(input, outputs_stem)
        
#         return seg_mask

if __name__ == "__main__":
    
    from torchinfo import summary
    from fvcore import nn as fnn

    model = DrivableAndLaneSegHead(
        in_channels= 48,
        out_channels= 2
    )

    input = torch.randn(1, 48, 72, 128)

    model_inputs = (input, )
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops, max_depth=5))
    summary(model, input_data=input, depth=5)