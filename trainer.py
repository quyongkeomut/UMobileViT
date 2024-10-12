import torch
import tqdm
from datetime import datetime
import os
import gc
today = datetime.today()

formatted_today = today.strftime('%Y_%m_%d')

class Trainer:
    def __init__(self,
                model,
                criterion,
                optimizer,
                metrics,
                num_epochs,
                device,
                train_loader,
                val_loader,
                num_classes = [2, 2],
                out_path = "./weights"
                ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.d_metrics = metrics(num_classes = num_classes[0], device = device)
        self.l_metrics = metrics(num_classes = num_classes[1], device = device)
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out_path = os.path.join(out_path, formatted_today)

        os.makedirs(self.out_path, exist_ok=True)

    def run(self):
        for epoch in range(self.num_epochs):
            self.train(epoch=epoch)
            
            self.val(epoch=epoch)
            torch.cuda.empty_cache()

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        print(F"TRAINING PHASE EPOCH: {epoch}")
        with tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.train_loader:
                image_name = data[0]
                inputs = data[1].to(self.device)
                d_targets, l_targets = data[2]
                d_targets, l_targets = d_targets.to(self.device), l_targets.to(self.device)

                d_outputs, l_outputs = self.model(inputs)

                d_loss = self.criterion(d_outputs, d_targets)
                l_loss = self.criterion(l_outputs, l_targets)

                _loss = d_loss + l_loss

                self.d_metrics.update(d_outputs, d_targets)
                d_metrics = self.d_metrics.compute()
                # print(d_metrics)
                self.l_metrics.update(l_outputs, l_targets)
                l_metrics = self.l_metrics.compute()

                d_iou, d_dice = d_metrics["iou"].mean().item(), d_metrics["dice"].mean().item()
                l_iou, l_dice = l_metrics["iou"].mean().item(), l_metrics["dice"].mean().item()

                metrics = {
                    "iou" : (d_iou + l_iou)/2,
                    "dice" : (d_dice + l_dice)/2
                }

                _loss.backward()
                self.optimizer.step()

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item(), **metrics)
                pbar.update(1)  # Increment the progress bar

                save_path = os.path.join(self.out_path, "model.pt")
                torch.save(self.model, save_path)
                torch.cuda.empty_cache()
                gc.collect()

            print("Drivable Mtrics:")
            print(self.d_metrics)
            print()

            print("Lane Metrics:")
            print(self.l_metrics)
            print()
            self.d_metrics.reset()
            self.l_metrics.reset()
        print(f'Epoch {epoch+1} Loss: {total_loss/len(self.train_loader)}')
        print()

    def val(self, epoch):
        self.model.eval()
        total_loss = 0
        print(F"VALIDATION PHASE EPOCH: {epoch}")
        with tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.val_loader:
                image_name = data[0]
                inputs = data[1].to(self.device)
                d_targets, l_targets = data[2]
                d_targets, l_targets = d_targets.to(self.device), l_targets.to(self.device)

                d_outputs, l_outputs = self.model(inputs)

                d_loss = self.criterion(d_outputs, d_targets)
                l_loss = self.criterion(l_outputs, l_targets)

                _loss = d_loss + l_loss

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item())
                pbar.update(1)  # Increment the progress bar
                torch.cuda.empty_cache()
                gc.collect()
        print(f'Epoch {epoch+1} Loss: {total_loss/len(self.val_loader)}')
        print()

if __name__ == "__main__":
    
    print(formatted_today)