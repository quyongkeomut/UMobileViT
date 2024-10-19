from typing import Optional, Tuple, Callable

import torch
from torch import Tensor
from torch.nn import (
    Module,
    GroupNorm,
    Dropout,
)
from torch.nn.parameter import Parameter

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
        kernel_size = (1, 1)
        
        if not self._qkv_same_channels:
            self.q_proj_weight = Parameter(torch.empty((1, in_channels, *kernel_size), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((in_channels, self.k_channels, *kernel_size), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((in_channels, self.k_channels, *kernel_size), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
            self.in_proj_weight = Parameter(torch.empty((1 + 2*in_channels, in_channels, *kernel_size), **factory_kwargs))
        self.out_proj_weight = Parameter(torch.empty((in_channels, in_channels, *kernel_size), **factory_kwargs))
        
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
        Z = self._sa_block(input)
        
        return Z 
        
        
    def _sa_block(
        self,
        input: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.self_attn(
            input, 
            input,
            input,
        )
        Z = attn_output[0]
        Z = self.dropout_self_attn(Z)
        Z = self.norm_self_attn(Z + input)
        return Z
    
    
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
        
        # Self-attention part
        Z = self._sa_block(input) # (N, C, spatial, seq)
        
        # Cross-attention part
        Z = self._ca_block(Z, memory) # (N, C, spatial, seq)
        
        return Z 
   
   
    def _sa_block(
        self,
        input: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.self_attn(
            input, 
            input,
            input,
        )
        Z = attn_output[0]
        Z = self.dropout_self_attn(Z)
        Z = self.norm_self_attn(Z + input)
        return Z  
        
        
    def _ca_block(
        self,
        input: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.cross_attn(
            input, 
            memory,
            memory,
        )
        Z = attn_output[0]
        Z = self.dropout_cross_attn(Z)
        Z = self.norm_cross_attn(Z + input)
        return Z