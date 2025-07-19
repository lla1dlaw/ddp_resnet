import pretty_errors
import torch
import torch.multiprocessing as mp
import torch.distributed as td
from torch.distributed import init_process_group, destroy_process_group

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


def load_train_objs(dataset_name: str, batch_size: int):
    print(f"- Loading Dataset {dataset_name.upper()}...")
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size)  # load your dataset
    return train_loader, test_loader


def main(rank: int, world_size: int, save_every: int, total_epochs: int, dataset_name: str, batch_size: int, arch: str, activation: str, num_trials: int):
    ddp_setup(rank, world_size)
    print(f"- Starting Train Loop With {td.get_world_size()} GPUs in DDP\n")
    train_loader, test_loader = load_train_objs(dataset_name, batch_size)
    labels = [label for _, label in train_loader.dataset]
    num_classes = len(torch.tensor(labels).unique())

    for trial in range(num_trials):
        print(f"\n---- Starting Trial {trial} ----")
        print(f"- Initializing model...")
        model = ComplexResNet(arch, num_classes=num_classes, activation_function=activation)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        print(f"- Initializing Trainer...")
        trainer = Trainer(model, train_loader, test_loader, optimizer, rank, save_every, trial)
        trainer.train(total_epochs)
    print("- Training Compete.")
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-arch', '--architecture', type=str, default='WS', choices=['WS', 'DN', 'IB'], help=f"Type of architecture for your resnets.\nChoices: 'WS', 'DN', or 'IB'.\nDefaults to 'WS'")
    parser.add_argument('-act', '--activation', metavar='ACT', type=str, default='complex_cardioid', choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'],
                        help="Activation function for ComplexResNet.")
    parser.add_argument('--epochs', type=int, default=5, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=math.inf, help='How often to save a snapshot')
    parser.add_argument('--dataset', type=str, default='S1SLC_CVDL', help='Dataset to use for trainng.')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials to run the experiment for.')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args.save_every, args.epochs, args.dataset, args.batch_size, args.architecture, args.activation, args.trials), nprocs=world_size)

