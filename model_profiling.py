import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from model.umobilevit import UMobileViT
from model.seg_head import SegmentationHead

from torch.profiler import (
    profile, 
    record_function, 
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler
)

from experiments_setup.bdd.configs.bdd100k_backbone_config import BACKBONE_CONFIGS
import time

if __name__ == "__main__":
    # environment setup
    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # 
    # import warnings
    # warnings.filterwarnings("ignore")
    # 
    # torch.backends.cudnn.benchmark = True
    # try:
    #     torch.multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    
    
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
    input = torch.randn(size=(5, *img_size), device="cuda")
    model_inputs = (input, )


    #
    # check model fps
    #
    # with torch.inference_mode():
    #     while True:
    #         t1 = time.time()
    #         
    #         output = model(input)
    #     
    #         print("FPS:", 1/(time.time() - t1))
    
    
    #
    # check model parameters by torchinfo
    #
    # summary(model, input_data=model_inputs, depth=5)
    
    
    #
    # torch profiler
    #
    activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
    with profile(
        activities=activities, 
        schedule=schedule(wait=3, warmup=3, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler("./log/umobilevit"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        use_cuda=True,
    ) as prof:
        for step in range(3 + 3 + 5):
            prof.step()
            if step >= 3 + 3 + 5:
                break
            with record_function("model_inference"):
                model(input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))   
    