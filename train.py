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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)


    random.seed(seed)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Training args')

    parser.add_argument('--scale', type=float, default=0.5,required=False, help='Model scale')
    parser.add_argument('--epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for model')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint for coutinue training')

    args = parser.parse_args()

    alpha = args.scale
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    device = args.device
    check_point = args.ckpt
    set_seed(seed)

    out_channels = bdd100k_config.out_channel
    patch_size = bdd100k_config.patch_size
    otim = bdd100k_config.optimizer
    optim_args = bdd100k_config.optim_args

    model = UMobileViT(
        alpha=alpha,
        in_channels=3,
        out_channels=out_channels,
        patch_size=patch_size
    )

    if check_point :
        model.load_state_dict(torch.load(check_point, weights_only=True))
    model = model.to(device)
    

    optimizer = OPTIMIZERS[otim](model.parameters(), **optim_args)
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