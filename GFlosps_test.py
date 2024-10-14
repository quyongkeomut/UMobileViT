import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

from torchinfo import summary

from model.umobilevit import UMobileViT
from model.seg_head import SegmentationHead

if __name__ == "__main__":
    model = UMobileViT(alpha=0.5, patch_size=(2, 2), out_channels=[2,2])
    
    img_size = (3, 320, 640)
    input = torch.randn(size=(1, *img_size))
    input_shape = (1, *img_size)
    
    model_inputs = (input, )

    output = model(input)
    print(output[0].shape, output[1].shape)
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops, max_depth=3))
    
    summary(model, input_data=model_inputs, depth=5)

    
    # --segmentation head---
    # model = SegmentationHead(
    #     in_channels= 48,
    #     out_channels= 2
    # )
# 
    # input = torch.randn(1, 48, 72, 128)
# 
    # model_inputs = (input, )
    # flops = fnn.FlopCountAnalysis(model, model_inputs)
    # print(fnn.flop_count_table(flops, max_depth=5))
    # summary(model, input_data=input, depth=5)