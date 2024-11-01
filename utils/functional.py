from typing import Optional, Tuple, List

import torch
from torch import Tensor

from torch.nn.parameter import Parameter
from torch.nn.functional import (
    conv2d, 
    softmax, 
    relu,
    dropout)
from torch.nn.modules.utils import _pair

from torch.overrides import (
    handle_torch_function,
    has_torch_function
)

CONV_KWARGS = {
    "stride": _pair(1),
    "padding": _pair(0),
    "dilation": _pair(1),
    "groups": 1,
} 


def unfold_custom(
    input: Tensor, 
    kernel_size: int | Tuple[int, int],
) -> Tensor:
    r"""
    Custom funcional unfold operation

    Args:
        input (Tensor): Tensor to be unfolded
        kernel_size (int | Tuple[int, int]): Kernel size of unfold operation

    Returns:
        Tensor: unfolded version of input
        
    Shape:
        inputs:
            input: (N, C, H, W)
            kernel_size: (2, ) or None
        output: (N, C, P, S), where P is the area of kernel and S is the number of
            such kernels in feature maps.
    """
    # check integrity
    kernel_size = _pair(kernel_size)
    assert (
        input.dim() == 4
    ), f"Encoder block expected input have 4 dimentions, got {input.dim()}." 
    N, C, H, W = input.shape
    assert (
        H % kernel_size[0] == 0 and W % kernel_size[1] == 0
    ), f"Height and width of feature map must be divisible by patch sizes, got input_size is {H, W}"

    h, w = kernel_size[0], kernel_size[1]
    lh, lw = H//kernel_size[0], W//kernel_size[1]
    S = lh * lw
    P = kernel_size[0] * kernel_size[1]
        
    Z = input.view(N, C, lh, h, lw, w).contiguous() # (N, C, lh, h, lw, w)
    Z = Z.transpose(-1, 2).contiguous() # (N, C, w, h, lw, lh)
    Z = Z.view(N, C, P, S).contiguous() # (N, C, P, S)
    return Z


def fold_custom(
    input: Tensor,
    output_size: Tuple[int, int],
    kernel_size: int | Tuple[int, int],
) -> Tensor:
    r"""
    Custom funcional fold operation

    Args:
        input (Tensor): Tensor to be folded
        kernel_size (int | Tuple[int, int]): Kernel size of fold operation

    Returns:
        Tensor: folded version of input
        
    Shape:
        inputs:
            input: (N, C, P, S)
            kernel_size: (2, ) or None
        output: (N, C, H, W)
    """
    # check integrity
    kernel_size = _pair(kernel_size)
    assert (
        input.dim() == 4
    ), f"Encoder block expected input have 4 dimentions, got {input.dim()}." 
    N, C, P, S = input.shape
    H, W = output_size
    
    h, w = kernel_size[0], kernel_size[1]
    lh, lw = H//h, W//w
    
    Z = input.view(N, C, w, h, lw, lh).contiguous() # (N, C, w, h, lw, lh)
    Z = Z.transpose(-1, 2).contiguous() # (N, C, lh, h, lw, w)
    Z = Z.view(N, C, H, W).contiguous()
    
    return Z


def _separable_attn_shape_check(
    query: Tensor, 
    key: Tensor, 
    value: Tensor,
) -> None:
    r"""Verifies the expected shape for query, key, value for Separable Attention.
    
    Desired shapes:
        - query (Tensor): (N, Cq, spatial, seq)
        - key (Tensor): (N, Ck, spatial, seq)
        - value (Tensor): (N, Cv, spatial, seq)
    """
    
    assert (
        query.shape[2:] == key.shape[2:] and value.shape[2:] == key.shape[2:]
    ), "query, key and value do not have the same spatial dimention and dimention of sequence"


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. We take advantage of these
            identities for performance if they are present. Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(N, C, ...)` where C is the channel dimension
        - k: :math:`(N, C, ...)` where C is the channel dimension
        - v: :math:`(N, C, ...)` where C is the channel dimension
        - w: :math:`(1 + C * 2, C, 1, 1)` where C is the channel dimension. The last
            two dimentions are kernel sizes, this makes convolution operation equals
            to linear projection on each pixels.
        - b: :math:`1 + C * 2` where C is the channel dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    C = q.size(1)
    if k is v:
        if q is k:
            # self-attention
            proj = conv2d(q, w, b, **CONV_KWARGS) # (N, 1 + C*2, ...)
            return proj.tensor_split([1, C+1], dim=1) # q, k, v order
        else:
            # encoder-decoder attention
            w_q, w_kv = w.tensor_split([1, ])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.tensor_split([1, ])
            q_proj = conv2d(q, w_q, b_q, **CONV_KWARGS) # (N, 1, ...)
            kv_proj = conv2d(k, w_kv, b_kv, **CONV_KWARGS) # (N, C*2, ...)
            k_proj, v_proj = kv_proj.tensor_split([C, ], dim=1) # (N, C, ...)
            return q_proj, k_proj, v_proj
    else:
        w_q, w_k, w_v = w.tensor_split([1, C+1])
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.tensor_split([1, C+1])
        return conv2d(q, w_q, b_q, **CONV_KWARGS), conv2d(k, w_k, b_k, **CONV_KWARGS), conv2d(v, w_v, b_v, **CONV_KWARGS)
    

def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor],
    b_k: Optional[Tensor],
    b_v: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(N, Cq, Qdims...)` where Cq is the query channel dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(N, Ck, Kdims...)` where Ck is the key channel dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(N, Cv, Vdims...)` where Cv is the value channel dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Cq, Cq, 1, 1)`
        - w_k: :math:`(Cq, Ck, 1, 1)`
        - w_v: :math:`(Cq, Cv, 1, 1)`
        - b_q: :math:`(Cq)`
        - b_k: :math:`(Cq)`
        - b_v: :math:`(Cq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[N, Cq, Qdims...]`
         - k': :math:`[N, Cq, Kdims...]`
         - v': :math:`[N, Cq, Vdims...]`

    """
    Cq, Ck, Cv = q.size(1), k.size(1), v.size(1)
    assert w_q.shape == (
        Cq,
        Cq,
        1, 1
    ), f"expecting query weights shape of {(Cq, Cq, 1, 1)}, but got {w_q.shape}"
    assert w_k.shape == (
        Cq,
        Ck,
        1, 1
    ), f"expecting key weights shape of {(Cq, Ck, 1, 1)}, but got {w_k.shape}"
    assert w_v.shape == (
        Cq,
        Cv,
        1, 1
    ), f"expecting value weights shape of {(Cq, Cv, 1, 1)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Cq,
    ), f"expecting query bias shape of {(Cq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Cq,
    ), f"expecting key bias shape of {(Cq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Cq,
    ), f"expecting value bias shape of {(Cq,)}, but got {b_v.shape}"
    return conv2d(q, w_q, b_q, **CONV_KWARGS), conv2d(k, w_k, b_k, **CONV_KWARGS), conv2d(v, w_v, b_v, **CONV_KWARGS)


def separable_attention_forward(
    query: Tensor,
    key: Tensor, 
    value: Tensor,
    channels_to_check: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    dropout_p: float,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    training: bool = False,
    need_context_scores: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Forward method for Separable Attention.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Separable Self-attention for Mobile Vision Transformers" for more details.
        channels_to_check: total dimension of the features.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        need_context_scores: output context_scores. Default: `False`
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.


    Shape:
        Inputs:
        - query: :math:`(N, C, spatial, seq)` where  N is the batch size, C is the channels dimension, spatial and
            seq are the result of patch division. Specifically, spatial is the patch area and seq is the number of
            those patches in feature maps.
        - key: :math:`(N, C, spatial, seq)` where  N is the batch size, C is the channels dimension, spatial and
            seq are the result of patch division. Specifically, spatial is the patch area and seq is the number of
            those patches in feature maps.
        - value: :math:`(N, C, spatial, seq)` where  N is the batch size, C is the channels dimension, spatial and
            seq are the result of patch division. Specifically, spatial is the patch area and seq is the number of
            those patches in feature maps.

        Outputs:
        - attn_output: :math:`(N, C, spatial, seq)` where N is the batch size, C is the channel dimension.
        - context_scores: Only returned when ``need_context_scores=True``.
    
    """
    
    
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
    ) 
    
    if has_torch_function(tens_ops):
        return handle_torch_function(
            separable_attention_forward,
            tens_ops,
            query,
            key,
            value,
            channels_to_check,
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            dropout_p,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            use_separate_proj_weight=use_separate_proj_weight,
            training=training,
            need_context_scores=need_context_scores,
        )
    #
    # check intergrity
    #
    
    # check the query, key and value if they have valid shapes
    _separable_attn_shape_check(query, key, value)
    
    # set up shape vars
    C = query.size(1)   
    
    assert (
        C == channels_to_check
    ), f"was expecting embedding dimension of {channels_to_check}, but got {C}."
     
    if not use_separate_proj_weight:
        assert (
            query.size(1) == key.size(1) and key.size(1) == value.size(1) 
        ), f"query's channels {query.size(1)}, key's channels {key.size(1)} and value's {value.size(1)} are not the same"
    
    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.tensor_split([1, C+1])
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )
    
    # prepare for attention
    context_scores = dropout(softmax(q, dim=-1),
                             p=dropout_p,
                             training=training) # (N, 1, spatial, seq_len)
    v = relu(v)
    
    
    #
    # (main part) compute Separable Attention and out projection
    #

    # compute context vector 
    context_vec = torch.sum(torch.mul(context_scores, k),
                            dim=-1, keepdim=True) # (N, C, spatial, 1)
    
    # compute attention values
    attn_values = torch.mul(context_vec, v) # (N, C, spatial, seq_len)
    attn_values = conv2d(attn_values, out_proj_weight, out_proj_bias, **CONV_KWARGS) # (N, Ci, spatial, seq_len)
    
    return (attn_values, context_scores) if need_context_scores else (attn_values, )