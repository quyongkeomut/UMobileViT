import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

from torchinfo import summary

from model.umobilevit import UMobileViT, UMobileViTDecoder

if __name__ == "__main__":
    model = UMobileViT(alpha=0.5, patch_size=(3, 2))
    img_size = (3, 360, 640)
    input = torch.randn(size=(1, *img_size))
    input_shape = (1, *img_size)
    
    model_inputs = (input, )
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops, max_depth=4))
    
    
    # --decoder---
    # alpha: float = 0.5
    # decoder = UMobileViTDecoder(alpha=alpha, patch_size=(3, 2))
    # decoder_inputs = ((torch.randn(size=(1, int(alpha*512), 45, 80)), 
    #                    torch.randn(size=(1, int(alpha*256), 90, 160)), 
    #                    torch.randn(size=(1, int(alpha*128), 180, 320))), 
    #                   )
    # flops = fnn.FlopCountAnalysis(decoder, decoder_inputs)
    # print(fnn.flop_count_table(flops, max_depth=4))