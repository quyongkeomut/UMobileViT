import tqdm

class Trainer:
    def __init__(self,
                model,
                criterion,
                optimizer,
                num_epochs,
                device,
                train_loader,
                val_loader
                ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def run(self):
        for epoch in range(self.num_epochs):
            self.train(epoch=epoch)
            self.val(epoch=epoch)

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

                _loss.backward()
                self.optimizer.step()

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item())
                pbar.update(1)  # Increment the progress bar

        print(f'Epoch {epoch+1} Loss: {total_loss/len(self.train_loader)}')
        print()

    def val(self, epoch):
        self.model.train()
        total_loss = 0
        print(F"VALIDATION PHASE EPOCH: {epoch}")
        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.train_loader:
                image_name = data[0]
                inputs = data[1].to(self.device)
                d_targets, l_targets = data[2]
                d_targets, l_targets = d_targets.to(self.device), l_targets.to(self.device)

                d_outputs, l_outputs = self.model(inputs)

                d_loss = self.criterion(d_outputs, d_targets)
                l_loss = self.criterion(l_outputs, l_targets)

                _loss = d_loss + l_loss

                _loss.backward()
                self.optimizer.step()

                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item())
                pbar.update(1)  # Increment the progress bar

        print(f'Epoch {epoch+1} Loss: {total_loss/len(self.train_loader)}')
        print()

