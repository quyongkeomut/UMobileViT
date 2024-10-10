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

from utils.transfomer import (
    TransformerEncoderLayer, 
    TransformerDecoderLayer,
    _get_initializer
)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([deepcopy(module) for i in range(N)])


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
            padding="same",
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
            self.global_block = _get_clones(transformer_block, num_transformer_block)
        else:
            self.global_block = ModuleList([])
        
        # local block is depthwise separable convolution, followed by a group norm layer
        self.local_block = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding="same",
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
        self.expansion_block = _get_expansion_block(
            in_channels,
            expansion_factor,
            norm_num_groups,
            bias,
            **factory_kwargs) if transformer_block is not None else Identity()
        
        # out normalization
        self.out_norm = GroupNorm(num_groups=norm_num_groups,
                                  num_channels=in_channels,
                                  **factory_kwargs)
        
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


class UMobileViTDecoderLayer(_UMobileViTLayer):
    def __init__(self, **kwargs) -> None:
        r"""
        
        Decoder layer of UMobileViT is made up of Transformer decoder blocks,
        local block made of convolutional layers, and out normalization layer.
        
        """
        transformer_block = TransformerDecoderLayer
        super().__init__(transformer_block, **kwargs)
        
        
    def forward(
        self, 
        input: Tensor,
        memory: Tensor
    ) -> Tensor:
        r"""
        Forward method of UMobileViT decoder layer

        Args:
            input (Tensor): input of decoder. Must be 4D Tensor.
            memory (Tensor): output of encoder layer at a specific stage. Must share the
                same spatial and sequence dimentions with input.

        Returns:
            Tensor: output of decoder.
            
        Shape
            Inputs:
                input: :math:`(N, C, H, W)`
                memory: :math:`(N, C, H, W)`

            Output: :math:`(N, C, H, W)`
        """
        # check intergrity: features sizes must be divisible by patch sizes respectively
        assert (
            input.dim() == 4
        ), f"Encoder block expected input have 4 dimentions, got {input.dim()}." 
        assert (
            memory.dim() == 4
        ), f"Encoder block expected memory have 4 dimentions, got {memory.dim()}." 
        
        N, C, *output_size = input.shape
        assert (
            output_size[0] % self.fold_params["kernel_size"][0] == 0 and output_size[1] % self.fold_params["kernel_size"][1] == 0
        ), f"Height and width of feature map must be divisible by patch sizes." 
        
        
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        
        # global block forward
        
        # unfolding
        Z = F.unfold(Z, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        mem = F.unfold(memory, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        
        for block in self.global_block: 
            Z = block(Z, mem)
            
        # folding
        Z = Z.view(N, C*self.patch_area, -1) # (N, C*P, S)
        Z = F.fold(Z, output_size=output_size, **self.fold_params) # (N, C, H, W)
        
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)
          

class DecoderOutLayer(_UMobileViTLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__(transformer_block=None, **kwargs)
        
        r"""
        
        Implement the custom global block. Instead of performing self-attention and 
        cross-attention, this global block will project the concatenated feature map
        to the model space.
        
        """
        
        self.global_block = ModuleList([
            Conv2d(
                in_channels=2*kwargs["in_channels"],
                out_channels=kwargs["in_channels"],
                kernel_size=1,
                bias=kwargs["bias"],
                device=kwargs["device"],
                dtype=kwargs["dtype"]),
            ReLU(),
            GroupNorm(
                num_groups=kwargs["norm_num_groups"],
                num_channels=kwargs["in_channels"],
                device=kwargs["device"],
                dtype=kwargs["dtype"]),
            Conv2d(
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["in_channels"],
                kernel_size=3,
                padding="same",
                groups=kwargs["in_channels"],
                bias=kwargs["bias"],
                device=kwargs["device"],
                dtype=kwargs["dtype"]),
            ReLU(),
            GroupNorm(
                num_groups=kwargs["norm_num_groups"],
                num_channels=kwargs["in_channels"],
                device=kwargs["device"],
                dtype=kwargs["dtype"])
        ])
        
    
    def _reset_parameters(self) -> None:
        super()._reset_parameters()
        for layer in self.global_block:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
    
    
    def forward(
        self, 
        input: Tensor, 
        memory: Tensor) -> Tensor:
        r"""
        Forward method of UMobileViT out decoder layer

        Args:
            input (Tensor): input of decoder. Must be 4D Tensor.
            memory (Tensor): output of encoder layer at a specific stage. Must share the
                same spatial and sequence dimentions with input.

        Returns:
            Tensor: output of decoder.
            
        Shape
            Inputs:
                input: :math:`(N, C, H, W)`
                memory: :math:`(N, C, H, W)`

            Output: :math:`(N, C, H, W)`
        """
        # check intergrity: features sizes must be divisible by patch sizes respectively
        assert (
            input.dim() == 4
        ), f"Encoder block expected input have 4 dimentions, got {input.dim()}." 
        assert (
            memory.dim() == 4
        ), f"Encoder block expected memory have 4 dimentions, got {memory.dim()}."
        assert (
            input.size(1) == memory.size(1)
        ), f"Encoder block expected input and memory have the same channels, got {input.size(1)} and {memory.size(1)}"
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        # global block forward
        Z = torch.cat([Z, memory], dim=1)
        Z = self.global_block[0](Z)
        for layer in self.global_block[1:]:
            Z = layer(Z)
        
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


class UMobileViTDecoder(Module):
    def __init__(
        self,  
        d_model: int = 92,
        expansion_factor: float = 3,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        num_transformer_block: int = 2,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        r"""
        Decoder of UMobileViT

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
            d_model > 0
        ), f"d_model must be greater than zero, got d_model={d_model}"
        assert (
            num_transformer_block > 0 
        ), f"num_transformer_block must be greater than zero, got num_transformer_block={num_transformer_block}"
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        decoder_layer_kwargs = {
            "in_channels": d_model,
            "expansion_factor": expansion_factor,
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "num_transformer_block": num_transformer_block,
            "initializer": initializer,
        }
        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        self.initializer = _get_initializer(initializer)
        
        # first stage block - upsample factor = 2
        stage_1 = ModuleList([
            Upsample(**upsampling_kwargs),
            UMobileViTDecoderLayer(**decoder_layer_kwargs, **factory_kwargs),
            UMobileViTDecoderLayer(**decoder_layer_kwargs, **factory_kwargs),
        ])
        
        # second stage block - upsample factor = 4
        stage_2 = ModuleList([
            Upsample(**upsampling_kwargs),
            UMobileViTDecoderLayer(**decoder_layer_kwargs, **factory_kwargs),
        ])
        
        # out block - upsample factor = 8
        out_block = ModuleList([
            Upsample(**upsampling_kwargs),
            DecoderOutLayer(**decoder_layer_kwargs, **factory_kwargs)
        ])
        
        self.layers = ModuleList([
            stage_1,
            stage_2,
            out_block,
        ])
    
    
    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        r"""
        Forward method of decoder.

        Args:
            inputs (Tuple[Tensor]): features of each stage from encoder. Must be arranged
            backward, i.e feature of the first stage must be put in the last, and the second 
            stage is put in the second last, and so forth.

        Returns:
            Tensor: output of decoder
        """
        if len(inputs) - 1 != len(self.layers):
            raise ValueError(f"number of inputs is expected to be equal to {len(self.layers) + 1}, got {len(inputs)}")
        
        Z = inputs[0]
        for i, memory in enumerate(inputs[1:]): 
            Z = self.layers[i][0](Z) # upsample
            for l in self.layers[i][1:]: # cross attention
                Z = l(Z, memory)
        
        return Z
        
     
class UMobileViT(Module):
    def __init__(
        self,
        in_channels: int = 3,
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
        
    
    def forward(self, input: Tensor) -> Tensor:
        encoder_outputs = self.encoder(input)
        return self.decoder(tuple(reversed(encoder_outputs)))