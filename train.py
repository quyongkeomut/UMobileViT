import torch
import tqdm
from torch.utils.data import DataLoader
from datasets.bdd_datasets import BDD100KDataset
from loss.loss import TotalLoss
from evaluate import SegmentationMetric
from trainer import Trainer
from experiments_setup.bdd.dataset import bdd100k_config
from optimizer.optimizer import OPTIMIZERS
from model.umobilevit import UMobileViT
import numpy as np
import random

def set_seed(seed):
    # Đặt seed cho PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu sử dụng GPU

    # Đặt seed cho NumPy
    np.random.seed(seed)

    # Đặt seed cho Random
    random.seed(seed)

# Ví dụ sử dụng
set_seed(42)

if __name__ == "__main__":

    alpha = bdd100k_config.alpha
    out_channels = bdd100k_config.out_channel
    patch_size = bdd100k_config.patch_size

    num_epochs = bdd100k_config.num_epochs
    batch_size = bdd100k_config.batch_size
    lr = bdd100k_config.lr
    momentum = bdd100k_config.momentum
    nesterov = bdd100k_config.nesterov
    otim = bdd100k_config.optimizer
    device = bdd100k_config.device

    model = UMobileViT(
        alpha=alpha,
        in_channels=3,
        out_channels=out_channels,
        patch_size=patch_size
    )
    model = model.to(device)
    otim_args = {
        "lr": lr,
        # "momentum": momentum,
        # "nesterov": nesterov 
    }

    optimizer = OPTIMIZERS[otim](model.parameters(), **otim_args)
    print(optimizer)

    criterion = TotalLoss()

    train_ds = BDD100KDataset()
    val_ds = BDD100KDataset(valid=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metrics=SegmentationMetric,
        num_classes=out_channels,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    trainer.run()