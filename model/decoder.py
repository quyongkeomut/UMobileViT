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
    Upsample,
)
import torch.nn.functional as F

from torch.nn.init import zeros_

from model.transfomer import (
    TransformerDecoderLayer,
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
        
        N, C, *input_size = input.shape
        assert (
            input_size[0] % self.fold_params["kernel_size"][0] == 0 and input_size[1] % self.fold_params["kernel_size"][1] == 0
        ), f"Height and width of feature map must be divisible by patch sizes." 
        
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        
        # global block forward
        
        # unfolding
        # Z = F.unfold(Z, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        # Z = Z.contiguous()
        # mem = F.unfold(memory, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        # mem = mem.contiguous()
        
        Z = unfold_custom(Z, kernel_size=self.fold_params["kernel_size"])
        mem = unfold_custom(Z, kernel_size=self.fold_params["kernel_size"])
        
        for block in self.global_block: 
            Z = block(Z, mem)
            
        # folding
        # Z = Z.view(N, C*self.patch_area, -1).contiguous() # (N, C*P, S)
        # Z = F.fold(Z, output_size=output_size, **self.fold_params) #.contiguous() # (N, C, H, W)
        Z = fold_custom(Z, output_size=input_size, kernel_size=self.fold_params["kernel_size"])
        
        
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
        
        self.global_block = Sequential(
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
                padding=(1, 1),
                groups=kwargs["in_channels"],
                bias=kwargs["bias"],
                device=kwargs["device"],
                dtype=kwargs["dtype"]),
            ReLU(),
        )
        
    
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
        Z = self.global_block(Z)    
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)    


def _get_upsample_block(initializer, **kwargs) -> Sequential:
    block = Sequential(
        Upsample(
            scale_factor=kwargs["scale_factor"],
            mode=kwargs["mode"]
        ),
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
        GroupNorm(
            num_groups=kwargs["norm_num_groups"],
            num_channels=kwargs["out_channels"],
            device=kwargs["device"],
            dtype=kwargs["dtype"]
        )
    )
    
    initializer(block[1].weight)
    if block[1].bias is not None:
        zeros_(block[1].bias)
        
    return block


class UMobileViTDecoder(Module):
    def __init__(
        self,  
        d_model: int = 96,
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
        self.initializer = _get_initializer(initializer)
        
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
        decoder_layer = UMobileViTDecoderLayer
        
        # first stage block - upsample factor = 2
        stage_1 = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
            *_get_clones(
                decoder_layer, 
                N=2,
                **decoder_layer_kwargs, 
                **factory_kwargs
            ),
        ])
        
        # second stage block - upsample factor = 4
        stage_2 = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
            *_get_clones(
                decoder_layer, 
                N=2,
                **decoder_layer_kwargs, 
                **factory_kwargs
            ),
        ])
        
        # out block - upsample factor = 8
        out_block = ModuleList([
            _get_upsample_block(
                self.initializer,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
                **factory_kwargs
            ),
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