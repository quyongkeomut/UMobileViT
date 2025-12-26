import os
import time
import numpy as np
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from loss.loss import SegLoss, BDD100KLoss
from metrics.metrics import SegmentationMetric
from trainer import Trainer, BDD100KTrainer
from optimizer.optimizer import OPTIMIZERS
from augmentation.augmentation import CustomAug
from model.umobilevit import umobilevit


NUM_DEVICE = torch.cuda.device_count()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)


def ddp_setup(rank: int, world_size: int):
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


def main(
    rank: int,
    world_size: int,
    task: str,
    scale: float,
    num_epochs: int,
    batch_size: int,
    check_point: str,
):
    
    ddp_setup(rank, world_size)
    
    if task == "bdd100k":
        from datasets.bdd100k_datasets import BDD100KDataset
        from experiments_setup.bdd100k.backbone_config import BACKBONE_CONFIGS
        from experiments_setup.bdd100k.experiment_config import (
            OUT_CHANNELS, 
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = BDD100KDataset()
        VAL_DS = BDD100KDataset(valid=True)
        IS_PIN_MEMORY = True
        NUM_WORKERS = 2
        head = "dual"
    
    elif task == "bdd100k_2":
        from datasets.bdd100k_direct_datasets import BDD100KDataset
        from experiments_setup.bdd100k_direct.backbone_config import BACKBONE_CONFIGS
        from experiments_setup.bdd100k_direct.experiment_config import (
            OUT_CHANNELS, 
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = BDD100KDataset()
        VAL_DS = BDD100KDataset(valid=True)
        IS_PIN_MEMORY = True
        NUM_WORKERS = 2
        head = "dual"
    
    out_dir = os.path.join("./weights", task)

    # print(OUT_CHANNELS)
    # these hyperparams depend on the dataset / experiment
    otim = OPTIMIZER_NAME
    optim_args = OPTIM_ARGS

    # initialize the model and optimizer2
    model_kwargs = {
        "out_channels": OUT_CHANNELS,
        "alpha": scale,
        "device": rank,
        **BACKBONE_CONFIGS
    }

    model = umobilevit(
        head=head,
        **model_kwargs
    )

    optimizer = OPTIMIZERS[otim](
        model.parameters(), 
        **optim_args
    )
    # print(optimizer)

    # load check point...
    if check_point:
        # load checkpoint
        check_point = torch.load(check_point, weights_only=False)
        model.load_state_dict(check_point["model_state_dict"])
        
    last_epoch = 0
    lr_scheduler_increase = None
    lr_scheduler_cosine = None
        
    # compile model
    model.compile(fullgraph=True, backend="cudagraphs")    

    # setup dataloaders
    train_loader = DataLoader(
        TRAIN_DS, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=DistributedSampler(TRAIN_DS),
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=IS_PIN_MEMORY
    )
    
    # print(next(iter(train_loader))[0].shape)
    val_loader = DataLoader(
        VAL_DS, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=DistributedSampler(VAL_DS),
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=IS_PIN_MEMORY,
    )

    # call the trainer
    if task == "bdd100k" or "bdd100k_2":
        criterion = BDD100KLoss()
        trainer = BDD100KTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            metrics=SegmentationMetric,
            num_epochs=num_epochs,
            start_epoch=last_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=OUT_CHANNELS,
            out_dir= out_dir,
            lr_scheduler_increase=lr_scheduler_increase,
            lr_scheduler_cosine=lr_scheduler_cosine,
            gpu_id=rank
        )
    else:
        criterion = SegLoss()
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            metrics=SegmentationMetric,
            num_epochs=num_epochs,
            start_epoch=last_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=OUT_CHANNELS,
            out_dir=out_dir,
            lr_scheduler_increase=lr_scheduler_increase,
            lr_scheduler_cosine=lr_scheduler_cosine,
            gpu_id=rank
        )
        
    trainer.run()
    destroy_process_group()


if __name__ == "__main__":
    # environment setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
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

    parser.add_argument('--task', type=str, default="bdd100k", required=False, help='Task to train model, valid values are one of [bdd100k, bdd100k_2]')
    parser.add_argument('--pretrained_head', type=str, default="dual", required=False, help='Pretrained head config to init weight from pretrained model, valid values are [single, dual]')
    parser.add_argument('--scale', type=float, default=0.25, required=False, help='Model scale')
    parser.add_argument('--epochs', type=int, default=5, help='Num epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint for coutinue training')

    args = parser.parse_args()
    
    
    # setup model hyperparameters and training parameters
    task = args.task
    pretrained_head = args.pretrained_head
    scale = args.scale
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    check_point = args.ckpt
    
    set_seed(seed)
    
    world_size = NUM_DEVICE
    args = (
        world_size, 
        task, 
        scale, 
        num_epochs, 
        batch_size, 
        check_point
    )
    mp.spawn(main, args=args, nprocs=world_size)
