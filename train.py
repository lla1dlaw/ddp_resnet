import pretty_errors
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, 
    MulticlassF1Score, MulticlassAUROC
)


import os
import math

from models import RealResNet, ComplexResNet
from Trainer import Trainer
from Datasets import get_dataloaders


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    print("- Configuring DDP...")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs(dataset_name: str, batch_size: int, arch: str):
    print(f"- Loading Dataset {dataset_name.upper()}...")
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size)  # load your dataset
    print(f"- Initializing model...")
    model = ComplexResNet(arch, activation_function='complex_cardioid')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    return train_loader, test_loader, model, optimizer


def main(rank: int, world_size: int, save_every: int, total_epochs: int, dataset_name: str, batch_size: int, arch: str):
    ddp_setup(rank, world_size)
    train_loader, test_loader, model, optimizer = load_train_objs(dataset_name, batch_size, arch)
    print(f"- Initializing Trainer...")
    trainer = Trainer(model, train_loader, test_loader, optimizer, rank, save_every)
    print("- Straning train loop...\n")
    trainer.train(total_epochs)
    print("- Training Compete.")
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-arch', '--architecture', type=str, default='WS', choices=['WS', 'DN', 'IB'], help=f"Type of architecture for your resnets.\nChoices: 'WS', 'DN', or 'IB'.\nDefaults to 'WS'")
    parser.add_argument('-act', '--activation', metavar='ACT', type=str, default='crelu', choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'],
                        help="Activation function for ComplexResNet.")
    parser.add_argument('--epochs', type=int, default=5, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=math.inf, help='How often to save a snapshot')
    parser.add_argument('--dataset', type=str, default='S1SLC_CVDL', help='Dataset to use for trainng.')
    parser.add_argument('--batch_size', default=2048, type=int, help='Input batch size on each device (default: 1024)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args.save_every, args.epochs, args.dataset, args.batch_size, args.architecture), nprocs=world_size)
