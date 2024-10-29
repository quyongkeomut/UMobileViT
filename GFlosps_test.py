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
import time

if __name__ == "__main__":
    #
    # init model
    #
    model = UMobileViT(
        alpha=0.25, 
        patch_size=(2, 2), 
        out_channels=[2,2],
        device="cuda",
        **BACKBONE_CONFIGS
    )
    model.eval()
    
    # load weights from checkpoint
    # check_point = torch.load(r"weights\scale_0.25\2024_10_20_13_42\last.pth", weights_only=True)
    # model.load_state_dict(check_point["model_state_dict"])
    
    
    #
    # create dummy data for profiling
    # 
    img_size = (3, 320, 640)
    img_shape = (1, *img_size)
    input = torch.randn(size=(1, *img_size), device="cuda")
    model_inputs = (input, )


    #
    # check model fps
    #
    with torch.inference_mode():
        fps_list = []
        while True:
            t1 = time.time()
            
            output = model(input)

            t: float = 1/(time.time() - t1)
            print("FPS:", t)
            fps_list.append(t)
            if len(fps_list) == 1000: 
                import statistics
                print("avg FPS:", statistics.mean(fps_list))
                print("std FPS:", statistics.stdev(fps_list))
                break
            
    
    #
    # check model parameters by torchinfo
    #
    # summary(model, input_data=model_inputs, depth=5)


    # pytorch op counter - thop
    # macs, params = profile(model, inputs=model_inputs)
    # macs, params = clever_format([macs*2, params], "%.3f")
    # print(macs, params)
    
    
    # torch profiler 
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
    #     with record_function("model_inference"):
    #         model(input)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    