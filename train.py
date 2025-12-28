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
from utils import set_env, set_seed, set_ddp, get_arg_parser
from optimizer.optimizer import OPTIMIZERS
from augmentation.augmentation import CustomAug
from model.umobilevit import umobilevit


NUM_DEVICE = torch.cuda.device_count()


def main(
    rank: int,
    world_size: int,
    task: str,
    scale: float,
    num_epochs: int,
    batch_size: int,
    check_point: str,
    is_ddp: bool,
    seed: int,
):  
    # SET ENVIRONMENT
    if is_ddp:
        set_ddp(rank, world_size)
    set_env()
    set_seed(seed + rank)
    
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
        trainer = BDD100KTrainer
    else:
        criterion = SegLoss()
        trainer = Trainer
    trainer = trainer(
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
    trainer.run()
    if is_ddp:
        destroy_process_group()


if __name__ == "__main__": 
    set_env()
    # arguments parser
    args = get_arg_parser()
    # setup model hyperparameters and training parameters
    world_size = NUM_DEVICE
    args_tuple = (world_size, ) + tuple(args.values())
    if args["is_ddp"]:
        mp.spawn(main, args=args_tuple, nprocs=world_size)
    else:
        main(0, *args_tuple)
