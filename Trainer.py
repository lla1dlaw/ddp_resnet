"""An abstracted trainer that wraps a model in DDP and handles training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from rich.progress import Progress
import wandb

from torchmetrics.classification import MulticlassAccuracy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.num_classes = model.num_classes
        self.model = DDP(model,  device_ids=[gpu_id])

    def _run_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def _run_val_batch(self, inputs, targets):
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss.item(), outputs

    def _run_epoch(self, epoch, progress_bar, task_id):
        loss_total = 0
        top1_acc = MulticlassAccuracy(num_classes=self.num_classes, k=1)
        top5_acc = MulticlassAccuracy(num_classes=self.num_classes, k=5)
        self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, outputs = self._run_batch(inputs, targets)
            loss_total += loss
            top1_acc.update(outputs, targets)
            top5_acc.update(outputs, targets)
            progress_bar.update(task_id, description=f"Epoch {epoch+1} ", advance=1)

        epoch_loss = loss_total / len(self.train_data)
        return epoch_loss, top1_acc.compute().item(), top5_acc.compute().item()

    def validate(self, epoch: int):
        top1_acc = MulticlassAccuracy(num_classes=self.num_classes, k=1)
        top5_acc = MulticlassAccuracy(num_classes=self.num_classes, k=5)
        total_loss = 0

        self.model.eval()

        with torch.no_grad():
            for inputs, targets in self.validation_data:
                inputs = inputs.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                loss, outputs = self._run_val_batch(inputs, targets)
                top1_acc.update(outputs, targets)
                top5_acc.update(outputs, targets)
                total_loss += loss

        epoch_loss = total_loss / len(self.validation_data)
        return epoch_loss, top1_acc.compute().item(), top5_acc.compue().item()


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"epoch_{epoch}_checkpoint.pt"
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        run = wandb.init(
            entity="liamlaidlaw-boise-state-university",
            project="Real-Federated-ResNet",
            config={
                "architecture": "ResNet",
                "dataset": 'CIFAR10',
                "epochs": max_epochs,
            },
        )
        total_steps = max_epochs * len(self.train_data) 
        with Progress() as progress_bar:
            task = progress_bar.add_task(description="Epoch 1 ", total=total_steps)
            for epoch in range(max_epochs):
                epoch_loss, train_top1, train_top5 = self._run_epoch(epoch, progress_bar, task)
                val_loss, val_top1, val_top5 = self.validate(epoch)
                run.log({
                    "train loss": epoch_loss,
                    "train acc": train_top1,
                    "train top5 acc": train_top5,
                    "val loss": val_loss,
                    "val acc": val_top1,
                    "val top5 acc": val_top5,
                })
                if self.gpu_id == 0 and epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
                self.validate(epoch)
        run.finish()

