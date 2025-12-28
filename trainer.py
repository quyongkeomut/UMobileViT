from ast import Raise
from typing import Type, \
            Optional, \
            Dict, \
            Tuple, \
            Any, \
            List, \
            Callable 
from contextlib import nullcontext
import tqdm
import time
from datetime import datetime
import os
import gc
import csv
import glob


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torchvision.transforms.v2 import Resize, InterpolationMode

TODAY = datetime.today()


class BaseTrainer:
    AVG_WEIGHT: float = 0.99
    FORMATTED_TODAY = TODAY.strftime('%Y_%m_%d_%H_%M')
    METRIC_KEYS = ['Loss', 'Acc', 'IoU', 'mIoU']
    
    
    def __init__(
        self,
        is_ddp: bool,
        model: Type[nn.Module],
        criterion: Callable,
        optimizer,
        metrics,
        num_epochs: int,
        start_epoch: int = 0,
        train_loader = None,
        val_loader = None,
        num_classes: int | List[int] = 2,
        out_dir = "./weights",
        lr_scheduler_increase: Optional[LinearLR] = None,
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        r"""
        Abstract Base Trainer. The inherited trainer must implement the following methods:
        - _forward_pass: Define the forward pass for training and validation. It should return loss and metrics dictionary.
        - _get_best_metric: Define how to get the best metric from the metrics dictionary for monitoring.
        - _reset_metric: Define how to reset the metrics tracker after each epoch.

        Args:
            is_ddp (bool): Decide whether the training setting is DDP or normal criteria. 
            model: Model for training
            criterion (Callable): Loss function
            optimizer: Optimizer
            metrics: Metrics tracker
            num_epochs (int): Number of training epochs
            start_epoch (int, optional): Starting epoch. Defaults to 0.
            gpu_id (int): Device which the model will be trained on. Defaults to 0.
            train_loader: Train dataset dataloader
            val_loader: Validation dataset dataloader
            num_classes (list, optional): Number of output classes. Defaults to [2, 2].
            out_path (str, optional): Output path for weights. Defaults to "./weights".
            
        """
        # setup training configs
        self.is_ddp = is_ddp 
        self.gpu_id = gpu_id
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.resize = Resize((360, 640), interpolation=InterpolationMode.NEAREST)
        self.best_metric = 0
        
        # setup path to save results
        alpha = model.module.alpha if self.is_ddp else model.alpha
        self.out_path = os.path.join(out_dir, f"scale_{alpha}", self.FORMATTED_TODAY)
        os.makedirs(self.out_path, exist_ok=True)
        
        # setup model and ema model
        self.model: nn.Module = model.to(gpu_id)
        if self.is_ddp:
            self.model: DDP = DDP(self.model, device_ids=[self.gpu_id])
        if self.gpu_id == 0:
            self.ema_model = AveragedModel(
                model=self.model.module if self.is_ddp else self.model, 
                multi_avg_fn=get_ema_multi_avg_fn(self.AVG_WEIGHT)
            )
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer
        if lr_scheduler_increase is None: 
            self.lr_scheduler_increase = LinearLR(
                self.optimizer,
                start_factor=1/5,
                total_iters=5
            )
        else:
            self.lr_scheduler_increase = lr_scheduler_increase
        if lr_scheduler_cosine is None:
            self.lr_scheduler_cosine = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs-5,
                eta_min=1e-4
            )
        else:
            self.lr_scheduler_cosine = lr_scheduler_cosine
        
        # setup metrics tracker
        self.metrics = metrics
        self.num_classes = num_classes
        
        # setup dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader


    def run(self) -> None:
        r"""
        Perform fitting loop and validation
        
        """
        # On train begin
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            val_csv_path = os.path.join(self.out_path, "val_metrics.csv")
    
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch'] + self.METRIC_KEYS)
            with open(val_csv_path, mode='w+', newline='') as val_csvfile:
                val_writer = csv.writer(val_csvfile)
                val_writer.writerow(['Epoch'] + self.METRIC_KEYS)
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            # Valid
            val_metrics = self.val(epoch=epoch)                
            # Log results
            if self.gpu_id == 0:
                with open(train_csv_path, mode='a', newline='') as train_csvfile:
                    train_writer.writerow(train_metrics)
                    train_writer = csv.writer(train_csvfile)
                with open(val_csv_path, mode='a', newline='') as val_csvfile:
                    val_writer = csv.writer(val_csvfile)
                    val_writer.writerow(val_metrics)
            # lr schedulers step
            if epoch >= 5:
                self.lr_scheduler_cosine.step()
            else:
                self.lr_scheduler_increase.step()
            torch.cuda.empty_cache()

        # At the end of training... 
        # clear gradient
        self.model.zero_grad(set_to_none=True)
        if self.gpu_id == 0:
            # save EMA model
            save_path = os.path.join(self.out_path, f"ema_model.pth")
            torch.save({
                "num_epoch": self.num_epochs,
                "model_state_dict": self.ema_model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        train_csvfile.close()
        val_csvfile.close()
       
       
    def _on_train_epoch_begin(self, epoch: int) -> None:
        """
        This method will be called at the begining of each training epoch

        Args:
            epoch (int): The current training epoch.
        """
        if self.gpu_id == 0:
            print(F"TRAINING PHASE EPOCH: {epoch+1}")
        # clear gradient
        self.model.zero_grad(set_to_none=True)
        self.model.train()
    
    
    def _on_train_epoch_end(
        self, epoch: int, 
        running_loss: float, 
        metrics: Dict[str, Any]
    ) -> None:
        """
        This method will be called after one training epoch finish to save a snapshot of the model.

        Args:
            epoch (int): The current training epoch.
            running_loss (float): Running average of loss value.
            metrics (Dict[str, Any]): Metric dictionary that store metrics
        """

        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} DONE')
            print()
            # also save model state dict along with optimizer's and scheduler's at the end of every epoch
            os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
            save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")

            # os.makedirs(save_path, exist_ok=True)
            torch.save({
                # "epoch": epoch,
                "model_state_dict": self.model.module.state_dict() if self.is_ddp else self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                "loss": running_loss,
                "metrics": metrics
            }, save_path)
        else:
            time.sleep(0.1)
     
    
    def train(self, epoch: int) -> List[Any]:     
        """
        Method to perform one training epoch

        Args:
            epoch (int): Epoch index
        Returns:
            List[Any]: A list containing epoch number, loss, and metric values.
        """
        self._on_train_epoch_begin(epoch)
        running_loss: float = 0
            
        # Train loop
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for idx, data in enumerate(self.train_loader):
                _loss, metrics = self._forward_pass(data)
                
                # Optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)                
                
                # Update EMA and logging
                if self.gpu_id == 0:
                    # update EMA model 
                    self.ema_model.update_parameters(self.model.module if self.is_ddp else self.model)
                    
                    # update progress bar
                    running_loss = 0.9*running_loss + 0.1*_loss.detach().item() if idx > 0 else _loss.detach().item()
                    pbar.set_postfix(loss=f"{running_loss:.4f}", **metrics)
                    pbar.update(1)  # Increase the progress bar

                    # save this step for backup...
                    save_path = os.path.join(self.out_path, "last.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.module.state_dict() if self.is_ddp else self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                        "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)
                else:
                    time.sleep(0.1)
                
                # clear cache
                gc.collect()
        
        self._on_train_epoch_end(epoch, running_loss, metrics)
        logs = [epoch + 1, f"{running_loss:4f}"] + [metrics[key] for key in self.METRIC_KEYS[1:]]
        # reset metrics tracker after every training epoch
        self._reset_metric()
        return logs
    
    
    def _on_val_epoch_begin(self, epoch: int) -> None:
        """
        This method will be call before one epoch of validation.

        Args:
            epoch (int): Epoch index
        """
        self.model.eval()
        if self.gpu_id == 0:
            print(F"VALIDATION PHASE EPOCH: {epoch+1}")    


    def _on_val_epoch_end(
        self, 
        epoch: int, 
        running_loss: float, 
        metrics: Dict[str, Any]
    ) -> None:
        """
        This method will be called after finishing one epoch of validation, to log results and
        save best model based on its monitoring metric(s)

        Args:
            epoch (int): Epoch index
            running_loss (float): Running average of loss value.
            metrics (Dict[str, Any]): Metrics dictionary.
        """
        best_metric = self._get_best_metric(metrics)

        if self.gpu_id == 0:
            print(f'Epoch {epoch+1}, Loss: {running_loss:4f} DONE')
            print()
            # save the best model on IoU metric
            if best_metric >= self.best_metric:
                files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
                for file_path in files_to_delete:
                    os.remove(file_path)

                save_path = os.path.join(self.out_path, f"best_IoU_{round(best_metric, 4)}_epoch_{epoch+1}.pth")
                torch.save({
                    # "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict() if self.is_ddp else self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    "loss": running_loss,
                    "metrics": metrics
                    }, save_path)
                self.best_metric = best_metric
        else:
            time.sleep(0.1)
    
    
    @torch.no_grad()
    def val(self, epoch) -> List[Any]:
        """
        Method to perform one validation epoch

        Args:
            epoch (_type_): Epoch index.

        Returns:
            List[Any]: A list containing epoch number, loss, and metric values.
        """
        # On val epoch begin
        self._on_val_epoch_begin(epoch)
        running_loss = 0
            
        pb = tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for idx, data in enumerate(self.val_loader):
                _loss, metrics = self._forward_pass(data)
                
                if self.gpu_id == 0:
                    # update progress bar
                    running_loss = 0.9*running_loss + 0.1*_loss.detach().item() if idx > 0 else _loss.detach().item()
                    pbar.set_postfix(loss=f"{running_loss:.4f}", **metrics)
                    pbar.update(1)  # Increase the progress bar
                else:
                    time.sleep(0.05)
        
        self._on_val_epoch_end(epoch, running_loss, metrics)
        logs = [epoch + 1, f"{running_loss:4f}"] + [metrics[key] for key in self.METRIC_KEYS[1:]]
        # reset metrics tracker after every validating epoch
        self._reset_metric()
        return logs
    
    
    def _forward_pass(self, data: Any) -> Tuple[Tensor, Any]:
        """
        This method must be defined as the forward pass for training and validation.

        Args:
            data (Any): One batch of data from dataloader.

        Returns:
            Tuple[Tensor, Any]: Loss tensor and metrics dictionary.
        """
        raise NotImplementedError
    
    
    def _get_best_metric(self, metrics: Dict[str, Any]) -> float:
        """
        This method must be implemented to calculate the monitoring metric from metrics to perform
         model checkpointing.

        Args:
            metrics (Dict[str, Any]): Metrics dictionary.

        Returns:
            float: A float value representing the monitoring metric.
        """
        raise NotImplementedError


    def _reset_metric(self) -> None:
        """
        After calculating metrics, metric trackers must be reset before moving to the next epoch. 
        Depends on tasks, this method must be implement separately.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    

class BDD100KTrainer(BaseTrainer):
    AVG_WEIGHT: float = 0.999
    METRIC_KEYS = ['Loss', 'd_Acc', 'd_mAcc', 'd_IoU', 'd_mIoU', 'l_Acc', 'l_IoU', 'l_mIoU']
    
    def __init__(self, num_classes=[2, 2], *args, **kwargs) -> None:
        """
        Pre-defined trainer inherits from BaseTrainer for BDD100k Dataset. The trainer keeps track 2 
        segmentation metrics: drivable area segmentation and lane segmentation.

        Args:
            num_classes (list, optional): Number of classes for drivable area and lane segmentation, respectively.
                Defaults to [2, 2].
        """
        
        super().__init__(num_classes=num_classes, *args, **kwargs)
        # setup metrics trackers
        self.d_metrics = self.metrics(self.num_classes[0])
        self.l_metrics = self.metrics(self.num_classes[1])
        self.METRIC_CALLERS = {
            'd_Acc': self.d_metrics.pixelAccuracy,
            'd_mAcc': self.d_metrics.meanPixelAccuracy,
            'd_IoU': self.d_metrics.IntersectionOverUnion,
            'd_mIoU': self.d_metrics.meanIntersectionOverUnion,
            'l_Acc': self.l_metrics.lineAccuracy,
            'l_IoU': self.l_metrics.IntersectionOverUnion,
            'l_mIoU': self.l_metrics.meanIntersectionOverUnion
        }
    

    def _forward_pass(self, data) -> Tuple[Tensor, Dict[str, Any]]:
        # get data
        image_name = data[0]
        inputs = data[1].to(self.gpu_id)
        d_targets, l_targets = data[2]
        d_targets, l_targets = d_targets.to(self.gpu_id), l_targets.to(self.gpu_id)
        
        # compute output, loss and metrics
        d_outputs, l_outputs = self.model(inputs)
        _loss = self.criterion((d_outputs, l_outputs), (d_targets, l_targets))
        
        # Resize for benchmark
        d_outputs = self.resize(d_outputs)
        d_targets = self.resize(d_targets)
        l_outputs = self.resize(l_outputs)
        l_targets = self.resize(l_targets)
        
        # Convert to numpy
        d_outputs = torch.argmax(d_outputs, dim=1).cpu().detach().numpy()
        d_targets = torch.argmax(d_targets, dim=1).cpu().detach().numpy()
        l_outputs = torch.argmax(l_outputs, dim=1).cpu().detach().numpy()
        l_targets = torch.argmax(l_targets, dim=1).cpu().detach().numpy()
        self.d_metrics.addBatch(d_outputs, d_targets)
        self.l_metrics.addBatch(l_outputs, l_targets)
        
        # calculate metrics of each task     
        metrics = {
            metric_key: metric_caller() for metric_key, metric_caller in self.METRIC_CALLERS.items()
        }
        return _loss, metrics

    
    def _reset_metric(self):
        self.d_metrics.reset()
        self.l_metrics.reset()
        

    def _get_best_metric(self, metrics: Dict[str, Any]) -> float:
        d_mIoU = metrics["d_mIoU"]
        l_IoU = metrics["l_IoU"]
        return (d_mIoU + l_IoU) / 2
        

class Trainer(BaseTrainer):
    AVG_WEIGHT: float = 0.999
    METRIC_KEYS = ['Loss', 'Acc', 'IoU', 'mIoU']
    
    
    def __init__(self, num_classes: int = 2, *args, **kwargs) -> None:
        """
        Pre-defined trainer inherits from BaseTrainer for a generic segmentation dataset. 

        Args:
            num_classes (int, optional): Number of classes for segmentation. Defaults to 2.
        """
        super().__init__(num_classes=num_classes, *args, **kwargs)
        # setup metrics tracker
        self.metrics = self.metrics(self.num_classes)
        self.METRIC_CALLERS = {
            'Acc': self.metrics.pixelAccuracy,
            'IoU': self.metrics.IntersectionOverUnion,
            'mIoU': self.metrics.meanIntersectionOverUnion
        }
    
    
    def _forward_pass(self, data) -> Tuple[Tensor, Dict[str, Any]]:
        # get data
        inputs = data[0].to(self.gpu_id)
        targets = data[1]
        targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets = targets.permute(0, 3, 1, 2)
        targets = targets.to(self.gpu_id)
        
        # compute output, loss and metrics
        outputs = self.model(inputs)
        _loss = self.criterion(outputs, targets)
        
        # Convert to numpy
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets =  torch.argmax(targets, dim=1).cpu().detach().numpy()
        self.metrics.addBatch(outputs, targets)
        
        # calculate metrics of each task
        acc = self.metrics.pixelAccuracy()
        IoU = self.metrics.IntersectionOverUnion()
        mIoU = self.metrics.meanIntersectionOverUnion()
        metrics = {
            "mIoU" : mIoU,
            "IoU" : IoU,
            "Acc" : acc
        }
        return _loss, metrics
    
    
    def _reset_metric(self):
        self.metrics.reset()
        

    def _get_best_metric(self, metrics: Dict[str, Any]) -> float:
        return metrics["mIoU"]
    

if __name__ == "__main__":
    ...