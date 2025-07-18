"""An abstracted trainer that wraps a model in DDP and handles training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from rich.progress import Progress
import wandb

from torchmetrics import MulticlassAccuracy


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
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        outputs = self.model(source)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def _run_epoch(self, epoch, progress_bar, task_id):
        num_classes = len(self.train_data.dataset.classes)
        loss_total = 0
        top1_acc = MulticlassAccuracy(num_classes=num_classes, k=1)
        top5_acc = MulticlassAccuracy(num_classes=num_classes, k=5)
        self.train_data.sampler.set_epoch(epoch)

        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, outputs = self._run_batch(source, targets)
            _, preds = torch.max(outputs, 1)
            loss_total += loss
            num_correct += torch.sum(preds == targets.data)
            progress_bar.update(task_id, description=f"Epoch {epoch+1} ", advance=1)

        epoch_loss = loss_total / len(self.train_data)
        epoch_acc = num_correct.double() / len(self.train_data)

        return epoch_loss, epoch_acc

    def validate(self, epoch: int, num_classes: int):
        top1_acc = MulticlassAccuracy(num_classes=num_classes, k=1)
        top5_acc = MulticlassAccuracy(num_classes=num_classes, k=5)

        for source, targets in self.validation_data:
            


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
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
                epoch_loss, epoch_acc = self._run_epoch(epoch, progress_bar, task)
                run.log({"acc": epoch_acc, "loss": epoch_loss})
                if self.gpu_id == 0 and epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)

        run.finish()

