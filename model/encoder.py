from typing import Tuple, Callable

import torch
from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    ReLU,
    Conv2d,
    GroupNorm,
)
import torch.nn.functional as F

from torch.nn.init import zeros_

from model.transfomer import (
    TransformerEncoderLayer, 
    _get_initializer
)
from model.module import (
    _UMobileViTLayer,
    _get_clones
)
from utils.functional import (
    unfold_custom,
    fold_custom
)


class UMobileViTEncoderLayer(_UMobileViTLayer):
    def __init__(self, **kwargs) -> None:
        r"""
        
        Encoder layer of UMobileViT is made up of Transformer encoder blocks,
        local block made of convolutional layers, and out normalization layer.
        
        """
        transformer_block = TransformerEncoderLayer
        super().__init__(transformer_block, **kwargs)
    
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward method of UMobileViT encoder layer

        Args:
            input (Tensor): input of encoder. Must be 4D Tensor.

        Returns:
            Tensor: output of encoder.
            
        Shape
            Input: :math:`(N, C, H, W)`
            Output: :math:`(N, C, H, W)`
        """
        # check intergrity: features sizes must be divisible by patch sizes respectively
        assert (
            input.dim() == 4
        ), f"Encoder block expected input have 4 dimensions, got {input.dim()}." 
        N, C, *input_size = input.shape
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        
        # global block forward
        
        # unfolding
        Z = unfold_custom(Z, kernel_size=self.fold_params["kernel_size"])
        
        for block in self.global_block: 
            Z = block(Z)
        
        # folding
        Z = fold_custom(Z, output_size=input_size, kernel_size=self.fold_params["kernel_size"])
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)


def _get_downsample_block(initializer, **kwargs) -> Sequential:
    block = Sequential(
        Conv2d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            groups=kwargs["groups"],
            bias=kwargs["bias"],
            device=kwargs["device"],
            dtype=kwargs["dtype"]
        ),
        ReLU(),
    )
    
    initializer(block[0].weight)
    if block[0].bias is not None:
        zeros_(block[0].bias)
        
    return block


class UMobileViTEncoder(Module):
    def __init__(
        self, 
        in_channels: int = 3,
        d_model: int = 64,
        expansion_factor: float = 3,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        num_transformer_block: int = 2,
        device=None,
        dtype=None
    ) -> None:
        r"""
        Encoder of UMobileViT

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
            in_channels > 0 
        ), f"in_channels must be greater than zero, got in_channels={in_channels}"
        assert (
            d_model > 0
        ), f"d_model must be greater than zero, got d_model={d_model}"
        assert (
            num_transformer_block > 0 
        ), f"num_transformer_block must be greater than zero, got num_transformer_block={num_transformer_block}"
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        
        # stem block - output stride = 8
        stem_conv_kwargs = {
            "kernel_size": 3,
            "stride": 2,
            "padding": (1, 1),
            "bias": bias,
        }
        
        d_stem = max(16, d_model//8)
        self.stem_block = ModuleList([
            # /2
            Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=d_stem,
                    **stem_conv_kwargs,
                    **factory_kwargs
                ),
                ReLU(),
            ),
            
            # /4
            Sequential(
                Conv2d(
                    in_channels=d_stem,
                    out_channels=d_stem,
                    **stem_conv_kwargs,
                    **factory_kwargs
                ),
                ReLU(),
                GroupNorm(
                    num_groups=norm_num_groups,
                    num_channels=d_stem,
                    **factory_kwargs
                )
            ),
            
            # /8
            Sequential(
                Conv2d(
                    in_channels=d_stem,
                    out_channels=d_model,
                    **stem_conv_kwargs,
                    **factory_kwargs
                ),
                ReLU(),
                GroupNorm(
                    num_groups=norm_num_groups,
                    num_channels=d_model,
                    **factory_kwargs
                )
            )
        ])
        
        
        #
        # encoder blocks setup
        #
        encoder_layer_kwargs = {
            "in_channels": d_model,
            "expansion_factor": expansion_factor,
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "num_transformer_block": num_transformer_block,
            "initializer": initializer,
        }
        downsampling_block_kwargs = {
            "in_channels": d_model,
            "out_channels": d_model,
            "kernel_size": 3,
            "stride": 2,
            "padding": (1, 1),
            "groups": d_model,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
        }
        encoder_layer = UMobileViTEncoderLayer
        
        # first stage block - output stride = 16
        stage_1 = Sequential(
            _get_downsample_block(
                self.initializer, 
                **downsampling_block_kwargs, 
                **factory_kwargs
            ),
            *_get_clones(
                encoder_layer, 
                N=1,
                **encoder_layer_kwargs,
                **factory_kwargs
            ),
        )
        
        # second stage block - output stride = 32
        stage_2 = Sequential(
            _get_downsample_block(
                self.initializer, 
                **downsampling_block_kwargs, 
                **factory_kwargs
            ),
            *_get_clones(
                encoder_layer, 
                N=2,
                **encoder_layer_kwargs,
                **factory_kwargs
            ),
        )
        
        # third stage block - output stride = 64
        # at this stage, patch size would be 1
        encoder_layer_kwargs["patch_size"] = (1, 1)
        stage_3 = Sequential(
            _get_downsample_block(
                self.initializer, 
                **downsampling_block_kwargs, 
                **factory_kwargs
            ),
            *_get_clones(
                encoder_layer, 
                N=2,
                **encoder_layer_kwargs,
                **factory_kwargs
            ),
        )
        
        self.layers = ModuleList([
            stage_1,
            stage_2,
            stage_3,
        ])
        
        self._reset_parameters()
        
        
    def _reset_parameters(self) -> None:
        for layer in self.stem_block:
            for component in layer:
                if isinstance(component, Conv2d):
                    self.initializer(component.weight)
                    if component.bias is not None:
                        zeros_(component.bias)
        
        for layer in self.layers:
            for component in layer:
                if isinstance(component, Conv2d):
                    self.initializer(component.weight)
                    if component.bias is not None:
                        zeros_(component.bias)
       
            
    def forward(self, input: Tensor) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
        r"""
        Return features at each stages, including stem stage.

        Args:
            input (Tensor): input image tensor.

        Returns:
            Tuple[Tensor]: features at each stages of encoder.
        """
        stem_outputs = []
        stage_outputs = []
        
        Z = input
        for layer in self.stem_block[:-1]:
            Z = layer(Z)
            stem_outputs.append(Z)
        Z = self.stem_block[-1](Z)
        stage_outputs.append(Z)
        
        for layer in self.layers:
            Z = layer(Z)
            stage_outputs.append(Z)
            
        return tuple(stem_outputs), tuple(stage_outputs), 