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
    Identity
)

from torch.nn.modules.utils import _pair

from torch.nn.init import zeros_

from model.transfomer import (
    TransformerEncoderLayer, 
    TransformerDecoderLayer,
    _get_initializer
)

def _get_clones(module, N):
    return [deepcopy(module) for i in range(N)]


def _get_expansion_block(
    in_channels: int, 
    expansion_factor: float,
    norm_num_groups: int = 1,
    bias: bool = True,
    **factory_kwargs
) -> Sequential:
    expanded_channels: int = int(expansion_factor*in_channels)
    block = Sequential(
        Conv2d(
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            bias=bias,
            **factory_kwargs),
        ReLU(),
        GroupNorm(
            num_groups=norm_num_groups,
            num_channels=expanded_channels,
            **factory_kwargs),
        Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            padding=(1, 1),
            groups=expanded_channels,
            bias=bias,
            **factory_kwargs),
        ReLU(),
        GroupNorm(
            num_groups=norm_num_groups,
            num_channels=expanded_channels,
            **factory_kwargs), 
        Conv2d(
            in_channels=expanded_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=bias,
            **factory_kwargs),
    )
    return block


class _UMobileViTLayer(Module):
    def __init__(
        self, 
        transformer_block: TransformerEncoderLayer | TransformerDecoderLayer | None,
        in_channels: int, 
        expansion_factor: float,
        patch_size: int | Tuple[int, int],
        dropout_p: float = 0.1,
        norm_num_groups: int = 1,
        bias: bool = True,
        num_transformer_block: int = 1,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        
        r"""
        
        A building layer of UMobileViT is made up of Transformer encoder/decoder blocks,
        local block made of convolutional layers, and out normalization layer.
        
        Args:
            in_channels (int): Number of channels in the input feature.
            patch_size (int | Tuple[int, int]): Size of patch using in Separable Attention layers
            dropout_p (float, optional): Dropout value. Defaults to 0.1.
            norm_num_groups (int, optional): The choice of normalization using in Transformer
                encoder block. If norm_num_groups = 1, this is equivalent to Layer
                Normalization. Defaults to 1.
            bias (bool, optional): If True, feed forward layers will have learnable biases. 
                Defaults to True.
            num_transformer_encoder (int, optional): number of transformer encoder blocks. 
                Defaults to 1.
            initializer (str | Callable[[Tensor], Tensor], optional): The choice of parameter 
                initializer. Default is "he_uniform" aka kaiming_uniform_.
                
        """
        
        assert num_transformer_block > 0, f"num_transformer_block must be greater than 0, got {num_transformer_block}"
        if in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0,"
                f"Got in_channels={in_channels} instead."
            )
        patch_size = _pair(patch_size)
        assert (
            all(size > 0 for size in patch_size) 
        ), "patch size elements must be greater than zero"
        assert expansion_factor >= 1, f"expansion_factor must be greater or equal to 1, got {expansion_factor}"
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fold_params = {"kernel_size": patch_size, "stride": patch_size}
        self.patch_area = prod(patch_size)
        self.initializer = _get_initializer(initializer)
        
        # global block made of transformer blocks
        tranformer_block_kwargs = {
            "in_channels": in_channels, 
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "initializer": initializer
        }
        if transformer_block is not None:
            transformer_block = transformer_block(**tranformer_block_kwargs, **factory_kwargs)
            global_block = _get_clones(transformer_block, num_transformer_block)
            if isinstance(transformer_block, TransformerEncoderLayer):
                self.global_block = Sequential(*global_block)
            elif isinstance(transformer_block, TransformerDecoderLayer):
                self.global_block = ModuleList(global_block)
        else:
            self.global_block = ModuleList([])
        
        # local block is depthwise separable convolution, followed by a group norm layer
        self.local_block = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=(1, 1),
                bias=bias,
                groups=in_channels,
                **factory_kwargs),
            ReLU(),
            GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                **factory_kwargs
            ),
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1, 
                bias=bias,
                **factory_kwargs),
            ReLU(),
            GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                **factory_kwargs
            )
        )
        
        # expansion block implementation, inspired by MobileNetV2 block
        self.expansion_block = (
            _get_expansion_block(
                in_channels,
                expansion_factor,
                norm_num_groups,
                bias,
                **factory_kwargs)
            ) if transformer_block is not None else Identity()
        
        # out normalization
        self.out_norm = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            **factory_kwargs
        )
        
        self._reset_parameters()


    def _reset_parameters(self) -> None:
        for layer in self.local_block:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
        
        if not isinstance(self.expansion_block, Identity):
            for layer in self.expansion_block:
                if isinstance(layer, Conv2d):
                    self.initializer(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)
        