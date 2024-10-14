import torch
import tqdm
from datetime import datetime
import os
import gc
import csv
import glob

today = datetime.today()

formatted_today = today.strftime('%Y_%m_%d_%H_%M')

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
        self.d_metrics = metrics(num_classes[0])
        self.l_metrics = metrics(num_classes[1])
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.out_path = os.path.join(out_path, formatted_today)

        self.best_IoU = 0
        os.makedirs(self.out_path, exist_ok=True)

    def run(self):
        train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
        val_csv_path = os.path.join(self.out_path, "val_metrics.csv")

        with open(train_csv_path, mode='w+', newline='') as train_csvfile:
            train_writer = csv.writer(train_csvfile)
            train_writer.writerow(['Epoch', 'Loss', 'd_Acc', 'd_IOU', 'd_mIOU', 'l_Acc', 'l_IOU', 'l_mIOU'])

        with open(val_csv_path, mode='w+', newline='') as val_csvfile:
            val_writer = csv.writer(val_csvfile)
            val_writer.writerow(['Epoch', 'Loss', 'd_Acc', 'd_IOU', 'd_mIOU',  'l_Acc', 'l_IOU', 'l_mIOU'])
        
        for epoch in range(self.num_epochs):
            
            train_metrics = self.train(epoch=epoch)
            with open(train_csv_path, mode='a', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(train_metrics)
            
            val_metrics = self.val(epoch=epoch)
            with open(val_csv_path, mode='a', newline='') as val_csvfile:
                val_writer = csv.writer(val_csvfile)
                val_writer.writerow(val_metrics)
            torch.cuda.empty_cache()

        train_csvfile.close()
        val_csvfile.close()

    def train(self, epoch):
        self.criterion.running_loss = 0.
        self.model.train()
        total_loss = 0
        print(F"TRAINING PHASE EPOCH: {epoch+1}")

        with tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.train_loader:
                self.optimizer.zero_grad()
                image_name = data[0]
                inputs = data[1].to(self.device)
                d_targets, l_targets = data[2]
                d_targets, l_targets = d_targets.to(self.device), l_targets.to(self.device)

                d_outputs, l_outputs = self.model(inputs)

                d_loss = self.criterion(d_outputs, d_targets)
                l_loss = self.criterion(l_outputs, l_targets)

                _loss = d_loss + l_loss

                d_outputs = torch.argmax(d_outputs, dim=1).cpu().detach().numpy()
                d_targets = torch.argmax(d_targets, dim=1).cpu().detach().numpy()

                l_outputs = torch.argmax(l_outputs, dim=1).cpu().detach().numpy()
                l_targets = torch.argmax(l_targets, dim=1).cpu().detach().numpy()

                self.d_metrics.addBatch(d_outputs, d_targets)
                
                self.l_metrics.addBatch(l_outputs, l_targets)
                
                d_acc = self.d_metrics.pixelAccuracy()
                d_IOU = self.d_metrics.IntersectionOverUnion()
                d_mIOU = self.d_metrics.meanIntersectionOverUnion()


                l_acc = self.l_metrics.lineAccuracy()
                l_IOU = self.l_metrics.IntersectionOverUnion()
                l_mIOU = self.d_metrics.meanIntersectionOverUnion()

                metrics = {
                    "d_mIOU" : d_mIOU,
                    "d_IOU" : d_IOU,
                    "l_IOU" : l_IOU,
                    "l_acc" : l_acc
                }
                _loss.backward(retain_graph=True)
                self.optimizer.step()

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item(), **metrics)
                pbar.update(1)  # Increment the progress bar

                save_path = os.path.join(self.out_path, "last.pt")
                torch.save(self.model, save_path)
                torch.cuda.empty_cache()
                gc.collect()
                # break

        save_weights = os.path.join(self.out_path, "epochs")
        os.makedirs(save_weights, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_weights, f"epoch_{epoch+1}.pth"))

        self.d_metrics.reset()
        self.l_metrics.reset()

        loss = (total_loss/len(self.train_loader))
        print(f'Epoch {epoch+1} Loss: {loss:4f}')
        print()

        return [epoch + 1, f"{loss:4f}", f"{d_acc:4f}", f"{d_IOU:4f}", f"{d_mIOU:4f}", f"{l_acc:4f}", f"{l_IOU:4f}", f"{l_mIOU:4f}"]

    @torch.no_grad()
    def val(self, epoch):
        self.criterion.running_loss = 0.
        self.model.eval()
        total_loss = 0
        print(F"VALIDATION PHASE EPOCH: {epoch+1}")
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

                d_outputs = torch.argmax(d_outputs, dim=1).cpu().detach().numpy()
                d_targets = torch.argmax(d_targets, dim=1).cpu().detach().numpy()

                l_outputs = torch.argmax(l_outputs, dim=1).cpu().detach().numpy()
                l_targets = torch.argmax(l_targets, dim=1).cpu().detach().numpy()

                self.d_metrics.addBatch(d_outputs, d_targets)
                
                self.l_metrics.addBatch(l_outputs, l_targets)
                
                d_acc = self.d_metrics.pixelAccuracy()
                d_IOU = self.d_metrics.IntersectionOverUnion()
                d_mIOU = self.d_metrics.meanIntersectionOverUnion()


                l_acc = self.l_metrics.pixelAccuracy()
                l_IOU = self.l_metrics.IntersectionOverUnion()
                l_mIOU = self.l_metrics.meanIntersectionOverUnion()

                metrics = {
                    "d_mIOU" : d_mIOU,
                    "d_IOU" : d_IOU,
                    "l_IOU" : l_IOU,
                    "l_acc" : l_acc
                }

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item(), **metrics)
                pbar.update(1)  # Increment the progress bar
                torch.cuda.empty_cache()
                gc.collect()
                # break

        self.d_metrics.reset()
        self.l_metrics.reset()

        loss = (total_loss/len(self.val_loader))
        print(f'Epoch {epoch+1} Loss: {loss:4f}')
        print()
        
        best_IoU = (d_mIOU + l_IOU) /2
        
        if best_IoU > self.best_IoU:
            files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
            for file_path in files_to_delete:
                os.remove(file_path)

            save_path = os.path.join(self.out_path, f"best_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), save_path)

        return [epoch + 1, f"{loss:4f}", f"{d_acc:4f}", f"{d_IOU:4f}", f"{d_mIOU:4f}", f"{l_acc:4f}", f"{l_IOU:4f}", f"{l_mIOU:4f}"]

if __name__ == "__main__":
    
    print(formatted_today)