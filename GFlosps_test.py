import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

from torchinfo import summary

from model.umobilevit import UMobileViT
from model.seg_head import SegmentationHead

from thop import profile, clever_format
# from ptflops import get_model_complexity_info
# from torch.profiler import profile, record_function, ProfilerActivity
from experiments_setup.bdd.configs.bdd100k_backbone_config import BACKBONE_CONFIGS

if __name__ == "__main__":
    model = UMobileViT(alpha=0.25, 
                       patch_size=(2, 2), 
                       out_channels=[2,2],
                       device="cuda",
                       **BACKBONE_CONFIGS
                       )
    model = model.to("cuda")
    img_size = (3, 320, 640)
    input = torch.randn(size=(1, *img_size))
    input_shape = (1, *img_size)
    
    model_inputs = (input.to("cuda"), )

    output = model(input.to("cuda"))
    print(output[0], output[1])
    
    
    # fvcore
    # flops = fnn.FlopCountAnalysis(model, model_inputs)
    # print(fnn.flop_count_table(flops, max_depth=3))
    
    
    # torch summary
    # summary(model, input_data=model_inputs, depth=5)


    # pytorch op counter - thop
    macs, params = profile(model, inputs=model_inputs)
    macs, params = clever_format([macs*2, params], "%.3f")
    print(macs)
    
    
    # flops counter - ptflops
    # macs, params = get_model_complexity_info(model, img_size, as_strings=True, backend='pytorch',
    #                                          print_per_layer_stat=True, verbose=True)
    # print(f'Computational complexity: {macs}')
    # print(f'Number of parameters: {params}')
    
    
    # torch profiler
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
    #     with record_function("model_inference"):
    #         model(input)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    
    # --segmentation head---
    # model = SegmentationHead(
    #     in_channels= 48,
    #     out_channels= 2
    # )
# 
    # input = torch.randn(1, 48, 72, 128)
# 
    # model_inputs = (input.to("cuda"), )
    # flops = fnn.FlopCountAnalysis(model, model_inputs)
    # print(fnn.flop_count_table(flops, max_depth=5))
    # summary(model, input_data=input.to("cuda"), depth=5)