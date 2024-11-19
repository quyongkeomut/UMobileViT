from typing import (
    Tuple, 
    Callable,
    Optional
)

import torch
from torch import Tensor
from torch.nn import Module

from model.encoder import UMobileViTEncoder
from model.decoder import UMobileViTDecoder
from model.seg_head import DrivableAndLaneSegHead, SegHead

     
class UMobileViT(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | Tuple[int, int],
        d_model: int,
        expansion_factor: float,
        alpha: float,
        patch_size: int | Tuple[int, int],
        dropout_p: float,
        norm_num_groups: int,
        bias: bool, 
        num_transformer_block: int,
        initializer: str | Callable[[Tensor], Tensor],
        device=None,
        dtype=None
    ) -> None:
        r"""
        Initializer of UmobileViT model. 

        Args:
            in_channels (int, optional): Number of channels of input image. Defaults to 3.
            out_channels (int | Tuple[int, int], optional): Number of channels of output massk. 
                Defaults to (2, 2).
            d_model (int, optional): Dimensions / Number of feature maps of the whole model. 
                Defaults to 64.
            expansion_factor (float, optional): Expansion factor in Inverted Residual block. 
                Defaults to 3.
            alpha (float, optional): Controls the width of the model. Defaults to 1.
            patch_size (int | Tuple[int, int], optional): Size of patch using in Separable Attention module . 
                Defaults to (2, 2).
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
            norm_num_groups (int, optional): Control number of groups to be normalized. If
                ``norm_num_groups`` = 1, this is equivalent to LayerNorm. Defaults to 4.
            bias (bool, optional): If ``True``, add trainable bias to building blocks. Defaults to True.
            num_transformer_block (int, optional): Number of Transformer block used in encoder/decoder layer. 
                Defaults to 2.
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
        self.seg_head = DrivableAndLaneSegHead(
            out_channels=out_channels,
            **kwargs 
        )
    
    
    def forward(self, input: Tensor) -> Tensor | Tuple[Tensor, Tensor]:
        stem_outputs, stage_outputs = self.encoder(input)
        output_decoder = self.decoder(tuple(reversed(stage_outputs)))
        output = self.seg_head(output_decoder, tuple(reversed(stem_outputs)))
        return output


def umobilevit(
    head: Optional[str] = "dual",
    pretrained_head: Optional[str] = "dual", 
    weights_path: Optional[str] = None,
    in_channels: int = 3,
    out_channels: int | Tuple[int, int] = (2, 2),
    d_model: int = 64,
    expansion_factor: float = 3,
    alpha: float = 1,
    patch_size: int | Tuple[int, int] = (2, 2),
    dropout_p: float = 0.1,
    norm_num_groups: int = 4,
    bias: bool = True, 
    num_transformer_block: int = 2,
    initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
    device=None,
    dtype=None
) -> UMobileViT:
    
    # load architecture 
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "d_model": d_model,
        "expansion_factor": expansion_factor,
        "alpha": alpha,
        "patch_size": patch_size,
        "dropout_p": dropout_p,
        "norm_num_groups": norm_num_groups,
        "bias": bias, 
        "num_transformer_block": num_transformer_block,
        "initializer": initializer,
        "device": device,
        "dtype": dtype
    }
    model = UMobileViT(**kwargs)
    
    # load pretrained weights (if specified)
    if weights_path:
        if pretrained_head != "dual":
            kwargs = {
                # "in_channels": in_channels,
                "out_channels": out_channels,
                "d_model": int(d_model*alpha),
                "expansion_factor": expansion_factor,
                # "alpha": alpha,
                "patch_size": patch_size,
                "dropout_p": dropout_p,
                "norm_num_groups": norm_num_groups,
                "bias": bias, 
                "num_transformer_block": num_transformer_block,
                "initializer": initializer,
                "device": device,
                "dtype": dtype
            }
            model.seg_head = SegHead(**kwargs)
        check_point = torch.load(weights_path, weights_only=False)
        model.load_state_dict(check_point["model_state_dict"])
    
    # if task is not lane and drivable segmentation, change the head of model
    if head != "dual":
        kwargs = {
            # "in_channels": in_channels,
            "out_channels": out_channels,
            "d_model": int(d_model*alpha),
            "expansion_factor": expansion_factor,
            # "alpha": alpha,
            "patch_size": patch_size,
            "dropout_p": dropout_p,
            "norm_num_groups": norm_num_groups,
            "bias": bias, 
            "num_transformer_block": num_transformer_block,
            "initializer": initializer,
            "device": device,
            "dtype": dtype
        }
        model.seg_head = SegHead(**kwargs)
    
    return model