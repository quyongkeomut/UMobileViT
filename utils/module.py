from typing import Optional, Tuple, Callable
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
    Dropout,
    Upsample
)
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

from torch.nn.init import (
    xavier_normal_,
    xavier_uniform_,
    kaiming_normal_,
    kaiming_uniform_,
    zeros_
)

from utils.functional import separable_attention_forward

_INITIALIZERS = {"glorot_normal": xavier_normal_,
                "glorot_uniform": xavier_uniform_,
                "he_normal": kaiming_normal_,
                "he_uniform": kaiming_uniform_,
                "zeros": zeros_}


def _get_initializer(name: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    if isinstance(name, str):
        try:
            initializer = _INITIALIZERS[name]
            return initializer
        except KeyError as e:
            raise ValueError(
                f"Valid initializers are {list(_INITIALIZERS.keys())}",
                f"Got {name} instead."
            )
    return name


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([deepcopy(module) for i in range(N)])


class SeparableAttention(Module):
    def __init__(
        self,
        in_channels: int,
        dropout_p: float = 0.1,
        bias: bool = True,
        k_channels: int = None,
        v_channels: int = None,
        initializer: str | Callable[[Tensor], Tensor] = kaiming_uniform_,
        device=None,
        dtype=None
    ) -> None:
        r"""
        Allows the model to jointly attend to information from different     
        representation subspaces in a patch of feature map, or cross 
        patches.
        
        Args:
            - in_channels: Number of channels.
            - dropout_p: Dropout probability use in context scores.
            - bias: If ``True``, use learnable bias in in-projection and out-
                projection.
            - k_channels, v_channels: Channels of ``key`` and ``value`` respectively.
            - initializer: Parameter initializer. 
        """
             
        if in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0,"
                f"Got in_channels={in_channels} instead"
            )
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        
        self.in_channels = in_channels
        self.k_channels = k_channels if k_channels is not None else in_channels
        self.v_channels = v_channels if v_channels is not None else in_channels
        self._qkv_same_channels = self.k_channels == self.v_channels and in_channels == self.v_channels
        self.dropout_p = dropout_p
        self.initializer = _get_initializer(initializer)
        
        if not self._qkv_same_channels:
            self.q_proj_weight = Parameter(torch.empty((1, in_channels, 1, 1), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((in_channels, self.k_channels, 1, 1), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((in_channels, self.k_channels, 1, 1), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
            self.in_proj_weight = Parameter(torch.empty((1 + 2*in_channels, in_channels, 1, 1), **factory_kwargs))
        self.out_proj_weight = Parameter(torch.empty((in_channels, in_channels, 1, 1), **factory_kwargs))
        
        if bias:
            self.in_proj_bias = Parameter(torch.empty(1 + 2*in_channels, **factory_kwargs))
            self.out_proj_bias = Parameter(torch.empty(in_channels, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
            self.register_parameter('out_proj_bias', None)
        
        self._reset_parameters()
    
    
    def _reset_parameters(self) -> None:
        if self._qkv_same_channels:
            self.initializer(self.in_proj_weight)
        else:
            self.initializer(self.q_proj_weight)
            self.initializer(self.k_proj_weight)
            self.initializer(self.v_proj_weight)
        
        if self.in_proj_bias is not None:
            zeros_(self.in_proj_bias)
            zeros_(self.out_proj_bias)
    
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_context_scores: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        r"""
        Compute Separable Attention outputs using querry, key and value 
        Args:
            - query, key, value: Map a querry and a set of key-value pairs to an output.
            - need_context_scores: If ``True``, return context scores. Default is
                ``False``.
        
        Shape:
            Inputs:
            - query: (N, Cq, spatial, seq_len) where ``N`` is the batch size, ``Cq`` is 
                the number of channels, ``spatial`` is the spatial dimentions of patched 
                feature. The dimention ``seq_len`` is the sequence length.
            - key: (N, Ck, spatial, seq_len) where ``N`` is the batch size, ``Ck`` is 
                the number of channels, ``spatial`` is the spatial dimentions of patched 
                feature. The dimention ``seq_len`` is the sequence length.
            - value: (N, Cv, spatial, seq_len) where ``N`` is the batch size, ``Cv`` is 
                the number of channels, ``spatial`` is the spatial dimentions of patched 
                feature. The dimention ``seq_len`` is the sequence length.
            
            
            Outputs:
            - attn_output: (N, Cq, spatial, seq_len) - the result of separable attention.
            - context_scores: (N, 1, spatial, seq_len) if ``need_context_scores`` is 
                ``True``, else ``None``. 
        """
        
        #
        # check the input shapes
        #
        assert (
            query.dim() == 4
        ), f"Expected query have 4 dimentions, got {input.shape}."
        assert (
            key.dim() == 4
        ), f"Expected query have 4 dimentions, got {key.shape}."
        assert (
            value.dim() == 4
        ), f"Expected query have 4 dimentions, got {value.shape}."
        
        
        #
        # calculate Separable Attention outputs
        #        
        return separable_attention_forward(
            query=query,
            key=key,
            value=value,
            channels_to_check=self.in_channels,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            dropout_p=self.dropout_p,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            use_separate_proj_weight = not self._qkv_same_channels,
            training=self.training,
            need_context_scores=need_context_scores
        )
            

class TransformerEncoderLayer(Module):
    def __init__(
        self, 
        in_channels: int,
        dropout_p: float = 0.1,
        norm_num_groups: int = 1,
        bias: bool = True,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        r"""
        TransformerEncoderLayer is made up of self-attention layer.

        Args:
            in_channels (int): Number of channels in the input feature.
            dropout_p (float, optional): Dropout value. Defaults to 0.1.
            norm_num_groups (int, optional): The choice of normalization using in Transformer
                encoder block. If norm_num_groups = 1, this is equivalent to Layer
                Normalization. Defaults to 1.
            bias (bool, optional): If True, feed forward layers will have learnable biases. 
                Defaults to True.
            initializer (str | Callable[[Tensor], Tensor], optional): The choice of parameter 
                initializer. Default is "he_uniform" aka kaiming_uniform_.
        """
        if in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0,"
                f"Got in_channels={in_channels} instead."
            )
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        # Implementation of Self Attention part
        self.self_attn = SeparableAttention(
            in_channels=in_channels,
            dropout_p=dropout_p,
            bias=bias,
            initializer=initializer,
            **factory_kwargs
        )
        self.dropout_self_attn = Dropout(dropout_p)
        self.norm_self_attn = GroupNorm(num_groups=norm_num_groups, 
                                        num_channels=in_channels,
                                        **factory_kwargs)
    
    
    def forward(
        self,
        input: Tensor,
        need_context_scores: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        r"""
        Forward method for Transformer encoder layer

        Args:
            input (Tensor): input feature map to the Tranformer Encoder block
            need_context_scores (bool, optional): If ``True``, return context scores of 
                Global Separable Attention. Defaults to False.

        Returns:
            Tensor | Tuple[Tensor, Tensor]: Output of Attention operation and context 
                scores.
            
        Shape:
            Inputs: 
            - input: (N, C, spatial, seq) where N is the batch size, C is number of 
                channels; spatial and sequence are the result of unfold operation to
                form the sequence dimention for attention operation.
                          
            Outputs:
            - attn_output: (N, C, spatial, seq)
            - context_scores: (N, 1, spatial, seq)
        """
        # validate the shape of input  
        assert (
            input.dim() == 4
        ), f"Transformer encoder block expected input have 4 dimentions, got {input.dim()}." 
        
        # Self-attention part
        attn_output = self._sa_block(input, need_context_scores)
        Z = attn_output[0]
        
        return Z if not need_context_scores else (Z, attn_output[1])
        
        
    def _sa_block(
        self,
        input: Tensor,
        need_context_scores: bool
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.self_attn(
            input, 
            input,
            input,
            need_context_scores
        )
        Z = attn_output[0]
        Z = self.dropout_self_attn(Z)
        Z = self.norm_self_attn(Z + input)
        return (Z, ) if not need_context_scores else (Z, attn_output[1])
        

class TransformerDecoderLayer(Module):
    def __init__(
        self, 
        in_channels: int, 
        dropout_p: float = 0.1, 
        norm_num_groups: int = 1, 
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform", 
        device=None, 
        dtype=None
    ) -> None:
        """
        TransformerDecoderLayer is made up of self attention layer and cross attention layer.
        Args:
            in_channels (int): Number of channels in the input feature.
            dropout_p (float, optional): Dropout value. Defaults to 0.1.
            norm_num_groups (int, optional): The choice of normalization using in Transformer
                encoder block. If norm_num_groups = 1, this is equivalent to Layer
                Normalization. Defaults to 1.
            bias (bool, optional): If True, feed forward layers will have learnable biases. 
                Defaults to True.
            initializer (str | Callable[[Tensor], Tensor], optional): The choice of parameter 
                initializer. Default is "he_uniform" aka kaiming_uniform_.
        """
        if in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0,"
                f"Got in_channels={in_channels} instead."
            )
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
                
        # Implementation of Cross Attention part
        self.cross_attn = SeparableAttention(
            in_channels=in_channels,
            dropout_p=dropout_p,
            bias=bias,
            initializer=initializer,
            **factory_kwargs
        )
        self.dropout_cross_attn = Dropout(dropout_p)
        self.norm_cross_attn = GroupNorm(num_groups=norm_num_groups, 
                                   num_channels=in_channels,
                                   **factory_kwargs)
        
        
    def forward(
        self,
        input: Tensor,
        memory: Tensor,
        need_context_scores: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        r"""
        Forward method for Transformer decoder layer

        Args:
            input (Tensor): input feature map to the Tranformer decoder layer
            memory (Tensor): output of encoder layer at a specific stage. Must share the
                same spatial and sequence dimentions with input.
            need_context_scores (bool, optional): If ``True``, return context scores of 
                Global Separable Attention. Defaults to False.

        Returns:
            Tensor | Tuple[Tensor, Tensor, Tensor]: Output of Attention operation and context 
                scores.
            
        Shape:
            Inputs: 
            - input: (N, C, spatial, seq) where N is the batch size, C is number of 
                channels; spatial and sequence are the result of unfold operation to
                form the sequence dimention for attention operation.
            - memory: (N, C, spatial, seq) where N is the batch size, C is number of 
                channels; spatial and sequence are the result of unfold operation to
                form the sequence dimention for attention operation.
                          
            Outputs:
            - attn_output: (N, C, spatial, seq)
            - self_context_scores: (N, 1, spatial, seq)
            - cross_context_scores: (N, 1, spatial, seq)
        """
        # validate the shape of input and memory
        assert (
            input.dim() == 4
        ), f"Transformer decoder block expected input have 4 dimentions, got {input.dim()}."
        assert (
            memory.dim() == 4
        ), f"Transformer decoder block expected memory have 4 dimentions, got {memory.dim()}." 
        
        # Cross-attention part
        cross_attn_output = self._ca_block(input, memory, need_context_scores)  
        Z = cross_attn_output[0]  # (N, C, spatial, seq)
        
        return Z if not need_context_scores else (Z, cross_attn_output[1])
     
        
    def _ca_block(
        self,
        input: Tensor,
        memory: Tensor,
        need_context_scores: bool
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.cross_attn(
            input, 
            memory,
            memory,
            need_context_scores
        )
        Z = attn_output[0]
        Z = self.dropout_cross_attn(Z)
        Z = self.norm_cross_attn(Z + input)
        return (Z, ) if not need_context_scores else (Z, attn_output[1])
    

class _UMobileViTLayer(Module):
    def __init__(
        self, 
        transformer_block: TransformerEncoderLayer | TransformerDecoderLayer,
        in_channels: int, 
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
        assert num_transformer_block >= 1, f"num_transformer_block must be greater or equal to 1, got {num_transformer_block}"
        if in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0,"
                f"Got in_channels={in_channels} instead."
            )
        patch_size = _pair(patch_size)
        assert (
            all(size > 0 for size in patch_size) 
        ), "patch size elements must be greater than zero"
        
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
        transformer_block = transformer_block(**tranformer_block_kwargs, **factory_kwargs)
        self.global_block = _get_clones(transformer_block, num_transformer_block)
        
        # local block is depthwise separable convolution
        conv_configs = {
            "in_channels": in_channels,
            "out_channels": in_channels,
            "bias": bias
        }
        self.local_block = Sequential(
            Conv2d(kernel_size=3, 
                   padding="same",
                   groups=in_channels,
                   **conv_configs,
                   **factory_kwargs),
            ReLU(),
        )
        
        # out normalization
        self.out_norm = GroupNorm(num_groups=norm_num_groups,
                                  num_channels=in_channels,
                                  **factory_kwargs)
        
        self._reset_parameters()


    def _reset_parameters(self) -> None:
        self.initializer(self.local_block[0].weight)
        
        if self.local_block[0].bias is not None:
            zeros_(self.local_block[0].bias)


class UMobileViTEncoderLayer(_UMobileViTLayer):
    def __init__(self, 
        in_channels: int, 
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
        
        Encoder layer of UMobileViT is made up of Transformer encoder blocks,
        local block made of convolutional layers, and out normalization layer.
        
        """
        transformer_block = TransformerEncoderLayer
        super().__init__(
            transformer_block,
            in_channels,
            patch_size,
            dropout_p,
            norm_num_groups,
            bias,
            num_transformer_block,
            initializer,
            device,
            dtype
        )
    
    
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
        ), f"Height and width of feature map must be divisible by patch sizes." 
        
        #
        # local block forward
        #
        Z = self.local_block(input) # (N, C, H, W)
        
        
        #
        # global block forward
        #
        
        # unfolding
        Z = F.unfold(Z, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        for block in self.global_block: 
            Z = block(Z)
        # folding
        Z = Z.view(N, C*self.patch_area, -1) # (N, C*P, S)
        Z = F.fold(Z, output_size=output_size, **self.fold_params) # (N, C, H, W)
        
        return self.out_norm(Z + input)


class UMobileViTDecoderLayer(_UMobileViTLayer):
    def __init__(self, 
        in_channels: int, 
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
        
        Decoder layer of UMobileViT is made up of Transformer decoder blocks,
        local block made of convolutional layers, and out normalization layer.
        
        """
        transformer_block = TransformerDecoderLayer
        super().__init__(
            transformer_block,
            in_channels,
            patch_size,
            dropout_p,
            norm_num_groups,
            bias,
            num_transformer_block,
            initializer,
            device,
            dtype
        )
        
        
    def forward(
        self, 
        input: Tensor,
        memory: Tensor
    ) -> Tensor:
        """Forward method of UMobileViT decoder layer

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
        
        #
        # local block forward
        #
        Z = self.local_block(input) # (N, C, H, W)
        
        
        #
        # global block forward
        #
        
        # unfolding
        Z = F.unfold(Z, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        mem = F.unfold(memory, **self.fold_params).view(N, C, self.patch_area, -1) # (N, C, P, S)   
        
        for block in self.global_block: 
            Z = block(Z, mem)
            
        # folding
        Z = Z.view(N, C*self.patch_area, -1) # (N, C*P, S)
        Z = F.fold(Z, output_size=output_size, **self.fold_params) # (N, C, H, W)
        
        return self.out_norm(Z + input)


class MobileNetv2Block(Module):
    def __init__(
        self, 
        in_channels: int,
        expansion_factor: int,
        stride: int = 1,
        out_channels: Optional[int] = None,
        bias: bool = True,
        norm_num_groups: int = 1,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        r"""
        Abstract MobileNetV2 block, can be used for feed forward and down-
        sampling.
            
        Args:
            in_channels: Number of input channels of feature map.
            expansion_factor: Expansion factor that multiplied with in_channels 
                to compute number of expanded feature map.
            stride: Stride of the depthwise convolution layer. Default is 
                ``1``.
            out_channels: Number of channels of output feature map. Default is
                ``None``. 
            bias: If ``True``, add learnable bias.
            norm_num_groups: Control number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm.
            initializer: The choice of parameter initializer. Default is 
                ``he_uniform`` aka kaiming_uniform_.
                
        """
        
        if in_channels <= 0 or (out_channels is not None and out_channels <= 0): 
            raise ValueError(
                f"in_channels must be greater than 0, out_channels can be either None or greater than 0"
                f"Got in_channels={in_channels} and out_channels={out_channels} instead"
            )
        assert (
            expansion_factor > 1
        ), f"expansion_factor must be greater than 1, got expansion_factor={expansion_factor}"
        
        super().__init__()
        
        expanded_channels = int(expansion_factor * in_channels)
        factory_kwargs = {"device": device, "dtype": dtype}
        out_channels = out_channels if out_channels is not None else in_channels
        self.initializer = _get_initializer(initializer)
        
        # pointwise expand conv block
        self.expand_block = Sequential(
            Conv2d(in_channels=in_channels,
                   out_channels=expanded_channels,
                   kernel_size=1,
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            GroupNorm(num_groups=norm_num_groups,
                      num_channels=expanded_channels,
                      **factory_kwargs)
        )

        # depthwise conv block
        self.depthwise_block = Sequential(
            Conv2d(in_channels=expanded_channels,
                   out_channels=expanded_channels,
                   kernel_size=3,
                   stride=stride,
                   padding = 1 if stride > 1 else "same",
                   groups=expanded_channels,
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            GroupNorm(num_groups=norm_num_groups,
                      num_channels=expanded_channels,
                      **factory_kwargs)
        )
        
        # pointwise out conv block
        self.out_block = Conv2d(in_channels=expanded_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                bias=bias,
                                **factory_kwargs)
        
        self.out_norm = GroupNorm(num_groups=norm_num_groups,
                                  num_channels=out_channels,
                                  **factory_kwargs)
        
        self.is_residual = stride == 1 and out_channels == in_channels
        self._reset_parameters()
        
        
    def _reset_parameters(self, ) -> None:
        self.initializer(self.expand_block[0].weight)
        self.initializer(self.depthwise_block[0].weight)
        self.initializer(self.out_block.weight)
        if self.expand_block[0].bias is not None:
            zeros_(self.expand_block[0].bias)
            zeros_(self.depthwise_block[0].bias)
            zeros_(self.out_block.bias)
            
        
    def forward(self, input: Tensor) -> Tensor:
        r"""Forward method of MobileNetV2 block
        
            Args:
                input: input feature map to the MobileNetV2 block
            
            Shape:
                Inputs: 
                - input: (N, Cin, H, W) where N`` is the batch size, Cin is number of 
                    channels, ``H`` and``W`` are spatial dimentions of feature map.
                              
                Outputs:
                - output: (N, Cout, H, W)
        """
        Z = self.expand_block(input)
        Z = self.depthwise_block(Z)
        Z = self.out_block(Z)
        Z = Z + input if self.is_residual else Z
        return self.out_norm(Z)
        

class MobileNetBlock(Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        bias: bool = True,
        norm_num_groups: int = 1,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None
    ) -> None:
        r"""
        Abstract MobileNet block, can be used for feed forward and down-
        sampling.

        Args:
            in_channels (int): Number of input channels of feature map.
            out_channels (Optional[int], optional): Number of channels of output feature 
                map. Default to ``None``.
            stride (int, optional): Stride of the depthwise convolution layer. Defaults to 1.
            bias (bool, optional): If ``True``, add learnable bias. Defaults to True.
            norm_num_groups (int, optional): ontrol number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm.. Defaults to 1.
            initializer (str | Callable[[Tensor], Tensor], optional): The choice of parameter 
                initializer. Defaults to "he_uniform".
   
        """
        if in_channels <= 0 or (out_channels is not None and out_channels <= 0): 
            raise ValueError(
                f"in_channels must be greater than 0, out_channels can be either None or greater than 0"
                f"Got in_channels={in_channels} and out_channels={out_channels} instead"
            )
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        out_channels = out_channels if out_channels is not None else in_channels
        self.initializer = _get_initializer(initializer)
        
        # depthwise conv
        self.depthwise_conv = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding = 1 if stride > 1 else "same",
            groups=in_channels,
            bias=bias,
            **factory_kwargs
        )
        
        # pointwise conv
        self.pointwise_conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            **factory_kwargs
        )
        
        # normalization layer
        self.norm = GroupNorm(num_groups=norm_num_groups,
                              num_channels=out_channels,
                              **factory_kwargs)
        
        self.is_residual = stride == 1 and out_channels == in_channels
        self._reset_parameters()
        
        
    def _reset_parameters(self, ) -> None:
        self.initializer(self.depthwise_conv.weight)
        self.initializer(self.pointwise_conv.weight)
        if self.depthwise_conv.bias is not None:
            zeros_(self.depthwise_conv.bias)
            zeros_(self.pointwise_conv.bias)
    
    
    def forward(self, input: Tensor) -> Tensor:
        r"""Forward method of MobileNet block
        
            Args:
                input: input feature map to the MobileNet block
            
            Shape:
                input: (N, Cin, H, W) where ``N`` is the batch size, ``Cin`` is 
                    number of channels, ``H`` and``W`` are spatial dimentions of 
                    feature map.
                              
                output: (N, Cout, H, W)
        """
        Z = F.relu(self.depthwise_conv(input))
        Z = F.relu(self.pointwise_conv(Z))
        Z = Z + input if self.is_residual else Z
        return self.norm(Z)
        

class UMobileViTEncoder(Module):
    def __init__(
        self, 
        in_channels: int = 3,
        alpha: float = 1,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
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
            alpha >= 0 
        ), f"alpha must be greater than zero, got alpha={alpha}"
        
        patch_size = _pair(patch_size)
        assert (
            all(size > 0 for size in patch_size) 
        ), f"patch size elements must be greater than zero, got patch_size={patch_size}"
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        encoder_layer_kwargs = {
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "initializer": initializer,
        }
        downsampling_block_kwargs = {
            "stride": 2,
            "bias": bias,
            "norm_num_groups": norm_num_groups,
            "initializer": initializer,
        }
        self.initializer = _get_initializer(initializer)
        self.alpha = alpha
        
        # stem block - output stride = 2
        stem_in_channels: int = int(alpha * 32)
        stem_out_channels: int = int(alpha * 128)
        stem_block = Sequential(
            Conv2d(in_channels=in_channels,
                   out_channels=stem_in_channels,
                   kernel_size=3,
                   padding="same",
                   bias=bias,
                   **factory_kwargs),
            ReLU(),
            MobileNetBlock(in_channels=stem_in_channels,
                           out_channels=stem_out_channels,
                           **downsampling_block_kwargs,
                           **factory_kwargs)
        )
        
        # first stage block - output stride = 4
        stage_1_out_channels: int = int(alpha * 128)
        stage_1 = Sequential(
            MobileNetBlock(in_channels=stem_out_channels,
                           out_channels=stage_1_out_channels,
                           **downsampling_block_kwargs,
                           **factory_kwargs),
            MobileNetBlock(in_channels=stage_1_out_channels,
                           out_channels=stage_1_out_channels,
                           stride=1,
                           bias=bias,
                           norm_num_groups=norm_num_groups,
                           initializer=initializer,
                           **factory_kwargs)
        )
        
        # second stage block - output stride = 8
        stage_2_out_channels: int = int(alpha * 256)
        stage_2 = Sequential(
            MobileNetBlock(in_channels=stage_1_out_channels,
                           out_channels=stage_2_out_channels,
                           **downsampling_block_kwargs,
                           **factory_kwargs),
            UMobileViTEncoderLayer(in_channels=stage_2_out_channels,
                                   **encoder_layer_kwargs,
                                   **factory_kwargs)
        )
        
        # third stage block - output stride = 16
        stage_3_out_channels: int = int(alpha * 384)
        stage_3 = Sequential(
            MobileNetBlock(in_channels=stage_2_out_channels,
                           out_channels=stage_3_out_channels,
                           **downsampling_block_kwargs,
                           **factory_kwargs),
            UMobileViTEncoderLayer(in_channels=stage_3_out_channels,
                                   **encoder_layer_kwargs,
                                   **factory_kwargs)
        )
        
        # fourth stage block - output stride = 32
        stage_4_out_channels: int = int(alpha * 512)
        stage_4 = Sequential(
            MobileNetBlock(in_channels=stage_3_out_channels,
                           out_channels=stage_4_out_channels,
                           **downsampling_block_kwargs,
                           **factory_kwargs),
            UMobileViTEncoderLayer(in_channels=stage_4_out_channels,
                                   **encoder_layer_kwargs,
                                   **factory_kwargs)
        )
        
        self.layers = ModuleList([
            stem_block,
            stage_1,
            stage_2,
            stage_3,
            stage_4,
        ])
        
        self._reset_parameters()
        
        
    def _reset_parameters(self) -> None:
        self.initializer(self.layers[0][0].weight)
        if self.layers[0][0].bias is not None: 
            zeros_(self.layers[0][0].bias)
       
            
    def forward(self, input: Tensor) -> Tuple[Tensor]:
        r"""
        Return features at each stages, including stem stage.

        Args:
            input (Tensor): input image tensor.

        Returns:
            Tuple[Tensor]: features at each stages of encoder.
        """
        outputs = []
        Z = self.layers[0](input)
        for layer in self.layers[1:]:
            Z = layer(Z)
            outputs.append(Z)
        return tuple(outputs)


class UMobileViTDecoder(Module):
    def __init__(
        self,  
        alpha: float = 1,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
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
            alpha >= 0 
        ), f"alpha must be greater than zero, got alpha={alpha}"
        
        patch_size = _pair(patch_size)
        assert (
            all(size > 0 for size in patch_size) 
        ), f"patch size elements must be greater than zero, got patch_size={patch_size}"
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        decoder_layer_kwargs = {
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias,
            "initializer": initializer,
        }
        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        pointwise_kwargs = {
            "kernel_size": 1,
            "bias": bias
        }
        self.initializer = _get_initializer(initializer)
        self.alpha = alpha
        
        # first stage block - upsample factor = 2
        in_channels: int = int(alpha * 512)
        stage_1_out_channels: int = int(alpha * 384)
        stage_1 = ModuleList([
            Upsample(**upsampling_kwargs),
            Conv2d(in_channels=in_channels,
                   out_channels=stage_1_out_channels,
                   **pointwise_kwargs,
                   **factory_kwargs),
            UMobileViTDecoderLayer(in_channels=stage_1_out_channels,
                                   **decoder_layer_kwargs,
                                   **factory_kwargs)
        ])
        
        # second stage block - upsample factor = 4
        stage_2_out_channels: int = int(alpha * 256)
        stage_2 = ModuleList([
            Upsample(**upsampling_kwargs),
            Conv2d(in_channels=stage_1_out_channels,
                   out_channels=stage_2_out_channels,
                   **pointwise_kwargs,
                   **factory_kwargs),
            UMobileViTDecoderLayer(in_channels=stage_2_out_channels,
                                   **decoder_layer_kwargs,
                                   **factory_kwargs)
        ])
        
        # third stage block - upsample factor = 8
        stage_3_out_channels: int = int(alpha * 128)
        stage_3 = ModuleList([
            Upsample(**upsampling_kwargs),
            Conv2d(in_channels=stage_2_out_channels,
                   out_channels=stage_3_out_channels,
                   **pointwise_kwargs,
                   **factory_kwargs),
            UMobileViTDecoderLayer(in_channels=stage_3_out_channels,
                                   **decoder_layer_kwargs,
                                   **factory_kwargs)
        ])

        # out block - upsample factor = 32
        self.out_block = Sequential(
            Upsample(**upsampling_kwargs),
            Conv2d(in_channels=stage_3_out_channels,
                   out_channels=stage_3_out_channels,
                   kernel_size=3,
                   padding="same",
                   groups=stage_3_out_channels,
                   bias=bias,
                   **factory_kwargs),
            ReLU()
        )
        
        self.layers = ModuleList([
            stage_1,
            stage_2,
            stage_3,
        ])

        self._reset_parameters()
    
    
    def _reset_parameters(self) -> None:
        self.initializer(self.out_block[1].weight)
        self.initializer(self.layers[0][1].weight)
        self.initializer(self.layers[1][1].weight)
        self.initializer(self.layers[2][1].weight)
        
        if self.out_block[1].bias is not None: 
            zeros_(self.out_block[1].bias)
            zeros_(self.layers[0][1].bias)
            zeros_(self.layers[1][1].bias)
            zeros_(self.layers[2][1].bias)
                        
    
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
            Z = self.layers[i][1](Z) # pointwise conv
            Z = self.layers[i][2](Z, memory) # cross attention
            
        # out block
        return self.out_block(Z) 
        
     
class UMobileViT(Module):
    def __init__(
        self,
        in_channels: int = 3,
        alpha: float = 1,
        patch_size: int | Tuple[int, int] = 2,
        dropout_p: float = 0.1,
        norm_num_groups: int = 4,
        bias: bool = True, 
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
        super().__init__()        
        kwargs = {
            "alpha": alpha,
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias, 
            "initializer": initializer,
            "device": device,
            "dtype": dtype
        }
        self.encoder = UMobileViTEncoder(in_channels=in_channels, **kwargs)
        self.decoder = UMobileViTDecoder(**kwargs)
        
    
    def forward(self, input: Tensor) -> Tensor:
        encoder_outputs = self.encoder(input)
        return self.decoder(tuple(reversed(encoder_outputs)))