"""An abstracted trainer that wraps a model in DDP and handles training."""
from datetime import datetime
import torch
import torch.nn as nn
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
        trial:int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.num_classes = model.num_classes
        self.model = DDP(model,  device_ids=[gpu_id])
        self.trial = trial

    def _run_batch(self, inputs, targets, criterion):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        print(targets.shape)
        print(targets)
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def _run_val_batch(self, inputs, targets, criterion):
        outputs = self.model(inputs)
        loss = criterion(outputs, targets.squeeze())
        return loss.item(), outputs

    def _run_epoch(self, epoch, progress_bar, task_id):
        loss_total = 0
        top1_acc = MulticlassAccuracy(self.num_classes, top_k=1)
        top5_acc = MulticlassAccuracy(self.num_classes, top_k=5)
        criterion = nn.CrossEntropyLoss()
        self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        epoch_start = datetime.now()
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, outputs = self._run_batch(inputs, targets, criterion)
            loss_total += loss
            top1_acc.update(outputs, targets)
            top5_acc.update(outputs, targets)
            progress_bar.update(task_id, description=f"Epoch {epoch+1} ", advance=1)
        epoch_end = datetime.now()
        total_epoch_duration = epoch_end - epoch_start
        epoch_duration_seconds = total_epoch_duration.total_seconds()
        epoch_loss = loss_total / len(self.train_data)
        return epoch_loss, top1_acc.compute().item(), top5_acc.compute().item(), epoch_duration_seconds

    def validate(self):
        top1_acc = MulticlassAccuracy(self.num_classes, top_k=1)
        top5_acc = MulticlassAccuracy(self.num_classes, top_k=5)
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        self.model.eval()

        with torch.no_grad():
            for inputs, targets in self.validation_data:
                inputs = inputs.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                loss, outputs = self._run_val_batch(inputs, targets, criterion)
                top1_acc.update(outputs, targets)
                top5_acc.update(outputs, targets)
                total_loss += loss

        epoch_loss = total_loss / len(self.validation_data)
        return epoch_loss, top1_acc.compute().item(), top5_acc.compute().item()


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"epoch_{epoch}_checkpoint.pt"
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        run = wandb.init(
            entity="liamlaidlaw-boise-state-university",
            project="SAR_ComplexResNet",
            name=f"Trial_{self.trial}",
            config={
                "architecture": "ComplexResNet",
                "dataset": 'S1SLC_CVDL',
                "epochs": max_epochs,
            },
        )
        total_steps = max_epochs * len(self.train_data) 
        with Progress() as progress_bar:
            task = progress_bar.add_task(description="Epoch 1 ", total=total_steps)
            for epoch in range(max_epochs):
                epoch_loss, train_top1, train_top5, epoch_duration = self._run_epoch(epoch, progress_bar, task)
                val_loss, val_top1, val_top5 = self.validate()
                run.log({
                    "train loss": epoch_loss,
                    "train acc": train_top1,
                    "train top5 acc": train_top5,
                    "val loss": val_loss,
                    "val acc": val_top1,
                    "val top5 acc": val_top5,
                    "epoch_duration_sec": epoch_duration,
                })
                if self.gpu_id == 0 and epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
        run.finish()

