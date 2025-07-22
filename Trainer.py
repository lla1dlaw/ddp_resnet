"""An abstracted trainer that wraps a model in DDP and handles training."""
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import wandb
import contextlib
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        arch: str,
        dataset_name: str,
        num_classes: int,
        num_channels: int,
        train_data: DataLoader,
        validation_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        trial: int,
        polarization: str,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.num_channels = num_channels
        root_dataset = train_data.dataset.dataset if hasattr(train_data.dataset, 'dataset') else train_data.dataset
        self.class_names = root_dataset.classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.trial = trial
        self.polarization = polarization
        self.model = model
        
        base_model_name = model.__class__.__name__
        if base_model_name == "ComplexResNet":
            self.model_name = f"{base_model_name}-{arch}-{model.activation_function}"
        else:
            self.model_name = f"{base_model_name}-{arch}"
        self.model_name = f"{self.model_name}-{self.dataset_name}"
        
        self.results_dir = os.path.join('./results', self.model_name)
        self.trial_data_dir = os.path.join(self.results_dir, "trial_data")
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        if self.gpu_id == 0:
            print(f"\nClasses for dataset: {self.num_classes}")
            print(f"Input Data Channels: {self.num_channels}")
            print(f"Model Expected Input Channels: {model.input_channels}")
            print(f"Model Classes: {model.num_classes}\n")

        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model,  device_ids=[self.gpu_id])

    def _run_batch(self, inputs, targets, criterion):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), outputs

    def _run_epoch(self, epoch, progress_bar, task_id):
        top1_acc = MulticlassAccuracy(num_classes=self.num_classes, top_k=1).to(self.gpu_id)
        f1_score = MulticlassF1Score(num_classes=self.num_classes).to(self.gpu_id)
        criterion = nn.CrossEntropyLoss().to(self.gpu_id)
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()

        loss_total = 0
        epoch_start = datetime.now()
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            loss, outputs = self._run_batch(inputs, targets, criterion)
            probs = F.softmax(outputs, dim=1)
            loss_total += loss
            top1_acc.update(probs, targets)
            f1_score.update(probs, targets)

            if progress_bar is not None and self.gpu_id == 0:
                progress_bar.update(task_id, description=f"Epoch {epoch+1} ", advance=1)
        
        if self.scheduler:
            self.scheduler.step()

        epoch_end = datetime.now()
        epoch_duration_seconds = (epoch_end - epoch_start).total_seconds()
        epoch_loss = loss_total / len(self.train_data)
        
        return epoch_loss, top1_acc.compute().item(), f1_score.compute().item(), epoch_duration_seconds
    
    def _validate(self, data_loader: DataLoader):
        self.model.eval()
        all_local_preds, all_local_targets = [], []
        total_loss = 0
        criterion = nn.CrossEntropyLoss().to(self.gpu_id)

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.gpu_id, non_blocking=True)
                targets = targets.to(self.gpu_id, non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                all_local_preds.append(outputs)
                all_local_targets.append(targets)

        local_preds_tensor = torch.cat(all_local_preds)
        local_targets_tensor = torch.cat(all_local_targets)
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, local_preds_tensor, local_targets_tensor

    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.results_dir, "checkpoints", f'trial_{self.trial}')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"epoch_{epoch}_checkpoint.pt")
        ckp = self.model.module.state_dict()
        torch.save(ckp, file_path)

    def _save_best_model(self):
        save_path = os.path.join(self.results_dir, "checkpoints", f'trial_{self.trial}')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, "best_model.pt")
        ckp = self.model.module.state_dict()
        torch.save(ckp, file_path)
        if self.gpu_id == 0:
            print(f"New best model saved to {file_path}")

    def _save_dataframe(self, dataframe: pd.DataFrame):
        os.makedirs(self.trial_data_dir, exist_ok=True)
        file_path = os.path.join(self.trial_data_dir, f"trial_{self.trial}.csv")
        dataframe.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

    def _send_wandb_link(self, url: str):
        pass

    def train(self, max_epochs: int):
        run = None
        if self.gpu_id == 0:
            os.environ["WANDB_SILENT"] = "true"
            run = wandb.init(
                entity="liamlaidlaw-boise-state-university",
                project=self.model_name,
                name=f"Trial_{self.trial}_{datetime.now()}",
                config={ "architecture": self.model.module.__class__.__name__, "dataset": self.dataset_name, "epochs": max_epochs },
            )
            print(f"Starting Training for {self.model_name}")
            if run.url:
                print(f"View Run Stats here: {run.url}")
                self._send_wandb_link(run.url)
        
        total_steps = max_epochs * len(self.train_data)
        progress_columns = [
            TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(),
            TextColumn("[green]Val Acc: {task.fields[val_acc]}"), TextColumn("[magenta]Best Val Acc: {task.fields[best_val_acc]}"),
            TextColumn("[red]Val Loss: {task.fields[val_loss]}"),
        ]
        
        best_val_acc = 0.0

        progress_context = Progress(*progress_columns) if self.gpu_id == 0 else contextlib.nullcontext()
        with progress_context as progress_bar:
            task = None
            if self.gpu_id == 0:
                task = progress_bar.add_task(description="Epoch 1 ", total=total_steps, val_acc=" - ", best_val_acc=" - ", val_loss=" - ")

            for epoch in range(max_epochs):
                train_loss, train_acc, train_f1, epoch_duration = self._run_epoch(epoch, progress_bar, task)
                
                val_loss_local, val_preds_local, val_targets_local = self._validate(self.validation_data)
                
                gathered_preds = [torch.zeros_like(val_preds_local) for _ in range(self.world_size)]
                gathered_targets = [torch.zeros_like(val_targets_local) for _ in range(self.world_size)]
                dist.all_gather(gathered_preds, val_preds_local)
                dist.all_gather(gathered_targets, val_targets_local)
                
                gathered_losses = [0.0] * self.world_size
                dist.all_gather_object(gathered_losses, val_loss_local)

                if self.gpu_id == 0:
                    all_preds = torch.cat(gathered_preds)
                    all_targets = torch.cat(gathered_targets)
                    val_loss = sum(gathered_losses) / self.world_size
                    
                    probs = F.softmax(all_preds, dim=1).to(self.gpu_id)
                    val_acc = MulticlassAccuracy(self.num_classes, top_k=1).to(self.gpu_id)(probs, all_targets).item()
                    val_precision = MulticlassPrecision(self.num_classes).to(self.gpu_id)(probs, all_targets).item()
                    val_recall = MulticlassRecall(self.num_classes).to(self.gpu_id)(probs, all_targets).item()
                    val_f1 = MulticlassF1Score(self.num_classes).to(self.gpu_id)(probs, all_targets).item()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        self._save_best_model()

                    progress_bar.update(task, val_acc=f"{val_acc:.4f}", best_val_acc=f"{best_val_acc:.4f}", val_loss=f"{val_loss:.4f}")

                    run.log({
                        'Training Accuracy': train_acc, 'Validation Accuracy': val_acc,
                        'Training Loss': train_loss, 'Validation Loss': val_loss,
                        'Training F1-Score': train_f1, 'Validation F1-Score': val_f1,
                        'Validation Precision': val_precision, 'Validation Recall': val_recall,
                    })

                    metrics = {"epoch": [epoch+1], "train loss": [train_loss], "train acc": [train_acc], "train f1": [train_f1],
                               "val loss": [val_loss], "val acc": [val_acc], "val f1": [val_f1], 
                               "val precision": [val_precision], "val recall": [val_recall], "epoch_duration_sec": [epoch_duration]}
                    self._save_dataframe(pd.DataFrame(metrics))

                    if epoch % self.save_every == 0:
                        self._save_checkpoint(epoch)

        dist.barrier()
        
        # --- FINAL TESTING ONCE AT THE END ---
        best_model_path = os.path.join(self.results_dir, "checkpoints", f'trial_{self.trial}', "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.module.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{self.gpu_id}'))

            test_loss_local, test_preds_local, test_targets_local = self._validate(self.test_data)
            
            gathered_test_preds = [torch.zeros_like(test_preds_local) for _ in range(self.world_size)]
            gathered_test_targets = [torch.zeros_like(test_targets_local) for _ in range(self.world_size)]
            dist.all_gather(gathered_test_preds, test_preds_local)
            dist.all_gather(gathered_test_targets, test_targets_local)
            
            if self.gpu_id == 0:
                print("\nEvaluating best model on the complete test set...")
                all_test_preds = torch.cat(gathered_test_preds)
                all_test_targets = torch.cat(gathered_test_targets)

                # --- FIXED: Move tensors to CPU before passing to wandb ---
                cpu_preds = all_test_preds.cpu()
                cpu_targets = all_test_targets.cpu()

                run.log({
                    "test_confusion_matrix": wandb.plot.confusion_matrix(
                        probs=cpu_preds,
                        y_true=cpu_targets,
                        class_names=self.class_names
                    )
                })
                # --- END FIX ---
                print(f"Final test confusion matrix logged to W&B run.")
        
        if self.gpu_id == 0:
            if run:
                run.finish()
