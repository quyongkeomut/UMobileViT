import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

# mport torch_tensorrt

from torchinfo import summary

from model.umobilevit import UMobileViT

from experiments_setup.bdd.configs.bdd100k_backbone_config import BACKBONE_CONFIGS


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000


def test_fps(
    model,
    model_inputs,
    n_steps: int
):
    with torch.inference_mode():
        # a couple of warm-up runs
        model(*model_inputs)
        model(*model_inputs)

        fps_list = []
        while True:
            t: float = timed(lambda: model(*model_inputs))
            fps: float = 1/t
            print("FPS:", fps)
            fps_list.append(fps)
            if len(fps_list) == n_steps: 
                import statistics
                print("avg FPS:", statistics.mean(fps_list))
                print("std FPS:", statistics.stdev(fps_list))
                break


def test_GFLOPs_profiler(
    model,
    model_inputs
):
    from torch.profiler import profile, record_function, ProfilerActivity
    # a couple of warm-up runs
    model(*model_inputs)
    model(*model_inputs)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            model(*model_inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def test_GFLOPs_thop(
    model,
    model_inputs
):
    from thop import profile, clever_format
    # a couple of warm-up runs
    model(*model_inputs)
    model(*model_inputs)
    macs, params = profile(model, inputs=model_inputs)
    flops, params = clever_format([macs*2, params], "%.3f")
    print("FLOPs: ", flops, "Params: ", params)


def show_parameters(model):
    summary(model, input_data=model_inputs, depth=5)


if __name__ == "__main__":
    # environment setup
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCHDYNAMO_DYNAMIC_SHAPES"] = "0"
    
    # The flags below controls whether to allow TF32 on cuda and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import warnings
    warnings.filterwarnings("ignore")
    
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    
    # create dummy data for profiling
    img_size = (3, 320, 640)
    input = torch.randn(size=(1, *img_size), device="cuda")
    model_inputs = (input, )
    
    
    #
    # init model
    #
    model = UMobileViT(
        alpha=0.25, 
        patch_size=(2, 2), 
        out_channels=[2,2],
        device="cuda",
        **BACKBONE_CONFIGS
    ).eval()
    
    # load weights from checkpoint
    # check_point = torch.load(r"weights\scale_0.25\2024_10_20_13_42\last.pth", weights_only=True)
    # model.load_state_dict(check_point["model_state_dict"])
        
    
    # enable the below line when using NVIDIA GPU V100, A100, or H100
    model.compile(fullgraph=True, backend="cudagraphs")
    
    
    # check model fps
    test_fps(model, model_inputs, n_steps=1000)
            
    
    # check model parameters by torchinfo
    # show_parameters(model)


    # pytorch op counter - thop
    # test_GFLOPs_thop(model, model_inputs)
    
    
    # torch profiler 
    # test_GFLOPs_profiler(model, model_inputs)
    