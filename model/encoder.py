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
from model.module import _UMobileViTLayer


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
            input: :math:`(N, C, H, W)`
            output: :math:`(N, C, H, W)`
        """
        # check intergrity: features sizes must be divisible by patch sizes respectively
        assert (
            input.dim() == 4
        ), f"Encoder block expected input have 4 dimentions, got {input.dim()}." 
        N, C, *output_size = input.shape
        assert (
            output_size[0] % self.fold_params["kernel_size"][0] == 0 and output_size[1] % self.fold_params["kernel_size"][1] == 0
        ), f"Height and width of feature map must be divisible by patch sizes, output_size is {output_size}" 
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        
        # global block forward
        
        # unfolding
        Z = F.unfold(Z, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        for block in self.global_block: 
            Z = block(Z)
        # folding
        Z = Z.view(N, C*self.patch_area, -1) # (N, C*P, S)
        Z = F.fold(Z, output_size=output_size, **self.fold_params) # (N, C, H, W)
        
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)




class UMobileViTEncoder(Module):
    def __init__(
        self, 
        in_channels: int = 3,
        d_model: int = 92,
        expansion_factor: float = 4,
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
            "padding": 1,
            "groups": d_model,
            "bias": bias,
        }
        self.initializer = _get_initializer(initializer)
        
        # stem block - output stride = 5
        stem_block = Sequential(
            Conv2d(in_channels=in_channels,
                   out_channels=32,
                   kernel_size=3,
                   stride=1,
                   padding="same",
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            Conv2d(in_channels=32,
                   out_channels=d_model,
                   kernel_size=3,
                   stride=1,
                   padding="same",
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            Conv2d(in_channels=d_model,
                   out_channels=d_model,
                   kernel_size=5,
                   stride=5,
                   padding=(2, 2),
                   groups=d_model,
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            GroupNorm(num_channels=d_model, 
                      num_groups=norm_num_groups,
                      **factory_kwargs)
        )
        
        # first stage block - output stride = 2
        stage_1 = Sequential(
            Conv2d(**downsampling_block_kwargs, **factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs,**factory_kwargs),
        )
        
        # second stage block - output stride = 4
        stage_2 = Sequential(
            Conv2d(**downsampling_block_kwargs, **factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs,**factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs, **factory_kwargs),
        )
        
        # third stage block - output stride = 8
        stage_3 = Sequential(
            Conv2d(**downsampling_block_kwargs, **factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs, **factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs, **factory_kwargs),
            UMobileViTEncoderLayer(**encoder_layer_kwargs, **factory_kwargs),
        )
        
        self.layers = ModuleList([
            stem_block,
            stage_1,
            stage_2,
            stage_3,
        ])
        
        self._reset_parameters()
        
        
    def _reset_parameters(self) -> None:
        for layer in self.layers:
            for component in layer:
                if isinstance(component, Conv2d):
                    self.initializer(component.weight)
                    if component.bias is not None:
                        zeros_(component.bias)
       
            
    def forward(self, input: Tensor) -> Tuple[Tensor]:
        r"""
        Return features at each stages, including stem stage.

        Args:
            input (Tensor): input image tensor.

        Returns:
            Tuple[Tensor]: features at each stages of encoder.
        """
        outputs = []
        Z = input
        for layer in self.layers:
            Z = layer(Z)
            outputs.append(Z)
        return tuple(outputs)