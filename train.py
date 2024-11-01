import torch
import tqdm
from torch.utils.data import DataLoader
from datasets.bdd_datasets import (
    TRAIN_DS, 
    VAL_DS,
    IS_PIN_MEMORY,
    NUM_WORKERS
)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from loss.loss import TotalLoss
from evaluate import SegmentationMetric
from trainer import Trainer
from experiments_setup.bdd.configs.bdd100k_backbone_config import BACKBONE_CONFIGS
from experiments_setup.bdd.configs.bdd100k_experiment_config import (
    OUT_CHANNELS, 
    PATCH_SIZE,
    OPTIMIZER_NAME,
    OPTIM_ARGS
)
from optimizer.bdd100k_optim import OPTIMIZERS
from model.umobilevit import UMobileViT
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)


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
    torch.backends.cudnn.benchmark = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    

    # arguments parser
    import argparse

    parser = argparse.ArgumentParser(description='Training args')

    parser.add_argument('--scale', type=float, default=0.25,required=False, help='Model scale')
    parser.add_argument('--epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint for coutinue training')

    args = parser.parse_args()
    
    
    # setup model hyperparameters and training parameters
    alpha = args.scale
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    device = args.device
    check_point = args.ckpt
    set_seed(seed)

    # these hyperparams depend on the dataset / experiment
    out_channels = OUT_CHANNELS
    patch_size = PATCH_SIZE
    otim = OPTIMIZER_NAME
    optim_args = OPTIM_ARGS

    # initialize the model and optimizer
    model = UMobileViT(
        alpha=alpha,
        out_channels=out_channels,
        patch_size=patch_size,
        device=device,
        **BACKBONE_CONFIGS
    )
    optimizer = OPTIMIZERS[otim](
        model.parameters(), 
        **optim_args
    )
    print(optimizer)

    # load check point...
    if check_point:
        check_point = torch.load(check_point, weights_only=True)
        
        model.load_state_dict(check_point["model_state_dict"])
        # optimizer.load_state_dict(check_point["optimizer_state_dict"])
        
        # lr_scheduler_increase = LinearLR(
        #     optimizer,
        #     start_factor=1/5,
        #     total_iters=5
        # )
        # lr_scheduler_increase.load_state_dict(check_point["lr_increase_state_dict"])
        
        # lr_scheduler_cosine = CosineAnnealingLR(
        #     optimizer, 
        #     T_max=num_epochs-5,
        #     eta_min=1e-4
        # )
        # lr_scheduler_cosine.load_state_dict(check_point["lr_cosine_state_dict"])
        
        # last_epoch = check_point["epoch"]
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None
    else:
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None
        
    model = model.to(device)
    model.compile(fullgraph=True, backend="cudagraphs")    

    criterion = TotalLoss()

    # setup dataloaders
    train_loader = DataLoader(
        TRAIN_DS, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=IS_PIN_MEMORY
    )
    val_loader = DataLoader(
        VAL_DS, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=IS_PIN_MEMORY,
    )

    # call the trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metrics=SegmentationMetric,
        num_epochs=num_epochs,
        start_epoch=last_epoch,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=out_channels,
        lr_scheduler_increase=lr_scheduler_increase,
        lr_scheduler_cosine=lr_scheduler_cosine,
        device=device
    )

    trainer.run()