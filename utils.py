import os
import time
import numpy as np
import random
import argparse

import torch
from torch.distributed import init_process_group


def set_env():
    # environment setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCHDYNAMO_DYNAMIC_SHAPES"] = "0"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"

    # The flags below controls whether to allow TF32 on cuda and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # set deterministic algorithms for reproducibility
    torch.backends.cudnn.benchmark = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    
def set_seed(seed):
    """
    Set seed method for reproducibility 

    Args:
        seed (int): Input seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)


def set_ddp(rank: int, world_size: int):
    """
    Init DDP

    Args:
        rank (int): A unique identifier that is assigned to each process
        world_size (int): Total process in a group
    """
    # this machine coordinates the communication across all processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    if rank == 0:
        time.sleep(30)
        

def get_arg_parser():
    parser = argparse.ArgumentParser(description="UMobileViT Training")
    parser.add_argument('--task', type=str, default="bdd100k", required=False, help='task name for training model, valid values are one of [bdd100k, bdd100k_2]')
    parser.add_argument('--scale', type=float, default=1.0, help='scale for backbone model')
    # parser.add_argument('--pretrained_head', type=str, default="dual", required=False, help='Pretrained head config to init weight from pretrained model, valid values are [single, dual]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint for coutinue training')
    parser.add_argument('--is_ddp', action="store_true", help='Option for choosing training in DDP or normal training criteria')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    
    args, _ = parser.parse_known_args()
    return vars(args)    
    
    

    

