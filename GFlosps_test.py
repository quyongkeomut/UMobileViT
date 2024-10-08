import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

from torchinfo import summary

from utils.module import UMobileViT  

if __name__ == "__main__":
    model = UMobileViT(alpha=0.5)
    img_size = (3, 256, 512)
    input = torch.randn(size=(1, *img_size))
    model_inputs = (input, )
    flops = fnn.FlopCountAnalysis(model, model_inputs)
    print(fnn.flop_count_table(flops))
    # print(summary(model, input_data=input))