from typing import Type, Optional
from contextlib import nullcontext
import tqdm
from datetime import datetime
import os
import gc
import csv
import glob

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torchvision.transforms.v2 import Resize, InterpolationMode

today = datetime.today()

formatted_today = today.strftime('%Y_%m_%d_%H_%M')

class BDD100KTrainer:
    def __init__(
        self,
        model: Type[torch.nn.Module],
        criterion,
        optimizer,
        metrics,
        num_epochs: int,
        start_epoch: int = 0,
        train_loader = None,
        val_loader = None,
        num_classes = [2, 2],
        out_dir = "./weights",
        lr_scheduler_increase: Optional[LinearLR] = None,
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        r"""
        Trainer

        Args:
            model: model for training
            criterion: loss function
            optimizer: optimizer
            metrics: metrics tracker
            num_epochs: number of training epochs
            gpu_id: device which the model will be trained on
            train_loader: train dataloader
            val_loader: valid dataloader
            num_classes (list, optional): number of output classes. Defaults to [2, 2].
            out_path (str, optional): output path for weights. Defaults to "./weights".
            
        """
        # setup model and ema model
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.ema_model = AveragedModel(
            model=self.model.module, 
            multi_avg_fn=get_ema_multi_avg_fn(0.999)
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
        
        # setup metrics trackers
        self.d_metrics = metrics(num_classes[0])
        self.l_metrics = metrics(num_classes[1])
        
        # setup dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # setup training configs
        self.num_epochs = num_epochs
        self.out_path = os.path.join(out_dir, f"scale_{self.model.module.alpha}", formatted_today)
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.resize = Resize((360, 640), interpolation=InterpolationMode.NEAREST)
        self.best_IoU = 0
        os.makedirs(self.out_path, exist_ok=True)
        

    def run(self):
        r"""
        Perform fitting loop and validation
        
        """
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            val_csv_path = os.path.join(self.out_path, "val_metrics.csv")
    
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss', 'd_Acc', 'd_mAcc', 'd_IoU', 'd_mIoU', 'l_Acc', 'l_IoU', 'l_mIoU'])
    
            with open(val_csv_path, mode='w+', newline='') as val_csvfile:
                val_writer = csv.writer(val_csvfile)
                val_writer.writerow(['Epoch', 'Loss', 'd_Acc', 'd_mAcc', 'd_IoU', 'd_mIoU', 'l_Acc', 'l_IoU', 'l_mIoU'])
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            # Valid
            val_metrics = self.val(epoch=epoch)                
           
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
            
            # torch.cuda.empty_cache()

        # save EMA model
        if self.gpu_id == 0:
            save_path = os.path.join(self.out_path, f"ema_model.pth")
            torch.save({
                "num_epoch": self.num_epochs,
                "model_state_dict": self.ema_model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        
        # train_csvfile.close()
        # val_csvfile.close()
    
    
    def evaluate(self):
        r"""
        Perform evaluate step
        
        """
        if self.gpu_id == 0:
            eval_csv_path = os.path.join(self.out_path, "eval_metrics.csv")
    
            with open(eval_csv_path, mode='w+', newline='') as eval_csvfile:
                eval_writer = csv.writer(eval_csvfile)
                eval_writer.writerow(['Epoch', 'Loss', 'd_Acc', 'd_mAcc', 'd_IoU', 'd_mIoU', 'l_Acc', 'l_IoU', 'l_mIoU'])
        
        for epoch in range(1):
            # Valid
            val_metrics = self.val(epoch=epoch)                
           
            if self.gpu_id == 0:
                with open(eval_csv_path, mode='a', newline='') as eval_csvfile:
                    val_writer = csv.writer(eval_csvfile)
                    val_writer.writerow(val_metrics)
            
            # torch.cuda.empty_cache()

        
        # train_csvfile.close()
        # val_csvfile.close()
    

    def _run_one_step(self, data):
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
        d_acc = self.d_metrics.pixelAccuracy()
        d_mAcc = self.d_metrics.meanPixelAccuracy()
        d_IoU = self.d_metrics.IntersectionOverUnion()
        d_mIoU = self.d_metrics.meanIntersectionOverUnion()
        l_acc = self.l_metrics.lineAccuracy()
        l_IoU = self.l_metrics.IntersectionOverUnion()
        l_mIoU = self.l_metrics.meanIntersectionOverUnion()
        
        metrics = {
            "d_mIoU" : d_mIoU,
            "d_IoU" : d_IoU,
            "d_acc": d_acc,
            "d_mAcc": d_mAcc,
            "l_mIoU": l_mIoU,
            "l_IoU": l_IoU, 
            "l_acc" : l_acc
        }
        
        return _loss, metrics
    
    
    def train(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(F"TRAINING PHASE EPOCH: {epoch+1}")
            
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                # clear gradient
                self.model.zero_grad(set_to_none=True)
                _loss, metrics = self._run_one_step(data)
                
                # optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()

                # update ema model 
                self.ema_model.update_parameters(self.model.module)
                
                if self.gpu_id == 0:
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item(), **metrics)
                    pbar.update(1)  # Increase the progress bar

                    # save this step for backup...
                    save_path = os.path.join(self.out_path, "last.pth")

                    torch.save({
                        # "epoch": epoch,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        # "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                        # "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.train_loader))
        
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
            # also save model state dict along with optimizer's and scheduler's at the end of every epoch
            os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
            save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")

            # os.makedirs(save_path, exist_ok=True)
            torch.save({
                # "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                # "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                # "loss": loss,
                # "metrics": metrics
            }, save_path)

        d_acc = metrics["d_acc"]
        d_mAcc = metrics["d_mAcc"]
        d_IoU = metrics["d_IoU"]
        d_mIoU = metrics["d_mIoU"]
        l_acc = metrics["l_acc"]
        l_IoU = metrics["l_IoU"]
        l_mIoU = metrics["l_mIoU"]
        
        # reset metrics tracker after every training epoch
        self.d_metrics.reset()
        self.l_metrics.reset()

        return [epoch + 1, f"{loss:4f}", f"{d_acc:4f}", f"{d_mAcc:4f}", f"{d_IoU:4f}", f"{d_mIoU:4f}", f"{l_acc:4f}", f"{l_IoU:4f}", f"{l_mIoU:4f}"]


    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(F"VALIDATION PHASE EPOCH: {epoch+1}")
            
        pb = tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.val_loader:
                _loss, metrics = self._run_one_step(data)
                
                if self.gpu_id == 0:
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item(), **metrics)
                    pbar.update(1)  # Increase the progress bar
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.val_loader))
        
        d_acc = metrics["d_acc"]
        d_mAcc = metrics["d_mAcc"]
        d_IoU = metrics["d_IoU"]
        d_mIoU = metrics["d_mIoU"]
        l_acc = metrics["l_acc"]
        l_IoU = metrics["l_IoU"]
        l_mIoU = metrics["l_mIoU"]

        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()

            # save the best model on IoU metric
            current_IoU = (d_mIoU + l_IoU) / 2
            if current_IoU >= self.best_IoU:
                files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
                for file_path in files_to_delete:
                    os.remove(file_path)

                save_path = os.path.join(self.out_path, f"best_IoU_{round(current_IoU, 4)}_epoch_{epoch + 1}.pth")
                torch.save({
                    # "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    # "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                    # "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    # "loss": loss,
                    # "metrics": metrics
                    }, save_path)

                self.best_IoU = current_IoU

        # reset metrics tracker after every validating epoch
        self.d_metrics.reset()
        self.l_metrics.reset()
        
        return [epoch + 1, f"{loss:4f}", f"{d_acc:4f}", f"{d_mAcc:4f}", f"{d_IoU:4f}", f"{d_mIoU:4f}", f"{l_acc:4f}", f"{l_IoU:4f}", f"{l_mIoU:4f}"]
    

class Trainer:
    def __init__(
        self,
        model: Type[torch.nn.Module],
        criterion,
        optimizer,
        metrics,
        num_epochs: int,
        start_epoch: int = 0,
        train_loader = None,
        val_loader = None,
        num_classes: int = 20,
        out_dir = "./weights",
        lr_scheduler_increase: Optional[LinearLR] = None,
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        r"""
        Trainer

        Args:
            model: model for training
            criterion: loss function
            optimizer: optimizer
            metrics: metrics tracker
            num_epochs: number of training epochs
            gpu_id: device which the model will be trained on
            train_loader: train dataloader
            val_loader: valid dataloader
            num_classes (int, optional): number of output classes. Defaults to 20.
            out_path (str, optional): output path for weights. Defaults to "./weights".
            
        """
        # setup model and ema model
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.ema_model = AveragedModel(
            model=self.model.module, 
            multi_avg_fn=get_ema_multi_avg_fn(0.999)
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
        
        # setup metrics trackers
        self.num_classes = num_classes
        self.metrics = metrics(num_classes)
        
        # setup dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # setup training configs
        self.num_epochs = num_epochs
        self.out_path = os.path.join(out_dir, f"scale_{self.model.module.alpha}", formatted_today)
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch

        self.best_IoU = 0
        os.makedirs(self.out_path, exist_ok=True)
        

    def run(self):
        r"""
        Perform fitting loop and validation
        
        """
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            val_csv_path = os.path.join(self.out_path, "val_metrics.csv")

            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss', 'Acc', 'IoU', 'mIoU'])

            with open(val_csv_path, mode='w+', newline='') as val_csvfile:
                val_writer = csv.writer(val_csvfile)
                val_writer.writerow(['Epoch', 'Loss', 'Acc', 'IoU', 'mIoU'])
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            # Valid
            val_metrics = self.val(epoch=epoch)
            
            # write results
            if self.gpu_id == 0:
                with open(train_csv_path, mode='a', newline='') as train_csvfile:
                    train_writer = csv.writer(train_csvfile)
                    train_writer.writerow(train_metrics)

                with open(val_csv_path, mode='a', newline='') as val_csvfile:
                    val_writer = csv.writer(val_csvfile)
                    val_writer.writerow(val_metrics)
            
            # lr schedulers step
            if epoch >= 5:
                self.lr_scheduler_cosine.step()
            else:
                self.lr_scheduler_increase.step()
            # break
            # torch.cuda.empty_cache()

        # save EMA model
        if self.gpu_id == 0:
            save_path = os.path.join(self.out_path, f"ema_model.pth")
            torch.save({
                "num_epoch": self.num_epochs,
                "model_state_dict": self.ema_model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        
        # train_csvfile.close()
        # val_csvfile.close()
        

    def _run_one_step(self, data):
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
    

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(F"TRAINING PHASE EPOCH: {epoch+1}")
            
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                # clear gradient
                self.model.zero_grad(set_to_none=True)
                _loss, metrics = self._run_one_step(data)
                
                # optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()
                
                # update ema model 
                self.ema_model.update_parameters(self.model.module)

                if self.gpu_id == 0:
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item(), **metrics)
                    pbar.update(1)  # Increase the progress bar

                    # save this step for backup...
                    save_path = os.path.join(self.out_path, "last.pth")

                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                        "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        acc = metrics["Acc"]
        IoU = metrics["IoU"]
        mIoU = metrics["mIoU"]
        
        # calculate averaged loss
        loss = (total_loss/len(self.train_loader))
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
            
            # also save model state dict along with optimizer's and scheduler's at the end of every epoch
            os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
            save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")

            # os.makedirs(save_path, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                "loss": loss,
                "metrics": metrics
                }, save_path)
        
        # reset metrics tracker after every training epoch
        self.metrics.reset()

        return [epoch + 1, f"{loss:4f}", f"{acc:4f}", f"{IoU:4f}", f"{mIoU:4f}"]


    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(F"VALIDATION PHASE EPOCH: {epoch+1}")
            
        pb = tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.val_loader:
                _loss, metrics = self._run_one_step(data)
                
                if self.gpu_id == 0:
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item(), **metrics)
                    pbar.update(1)  # Increase the progress bar
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.val_loader))
        
        acc = metrics["Acc"]
        IoU = metrics["IoU"]
        mIoU = metrics["mIoU"]
        
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
            
            # save the best model on IoU metric
            current_IoU = mIoU 
            if current_IoU >= self.best_IoU:
                files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
                for file_path in files_to_delete:
                    os.remove(file_path)

                save_path = os.path.join(self.out_path, f"best_IoU_{round(current_IoU, 4)}_epoch_{epoch + 1}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    "loss": loss,
                    "metrics": metrics
                    }, save_path)
                
                self.best_IoU = current_IoU

        # reset metrics tracker after every validating epoch
        self.metrics.reset()
        
        return [epoch + 1, f"{loss:4f}", f"{acc:4f}", f"{IoU:4f}", f"{mIoU:4f}"]

if __name__ == "__main__":
    
    print(formatted_today)