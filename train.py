import pretty_errors
import os
import math
import torch
from torch.distributed import init_process_group, destroy_process_group
from models import RealResNet, ComplexResNet
from Trainer import Trainer
from Datasets import get_dataloaders


def ddp_setup():
    """
    Sets up the distributed data parallel environment.
    Assumes that MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE are in the environment.
    """
    print("- Configuring DDP...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_train_objs(dataset_name: str, batch_size: int):
    print(f"- Loading Dataset {dataset_name.upper()}...")
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size)  
    return train_loader, test_loader


def main(rank: int, save_every: int, total_epochs: int, dataset_name: str, batch_size: int, arch: str, activation: str, num_trials: int):
    ddp_setup()
    if rank == 0:
        print(f"- Starting Train Loop on Rank {rank} with {torch.cuda.device_count()} GPUs in DDP\n")
    train_loader, test_loader = load_train_objs(dataset_name, batch_size)
    labels = [label for _, label in train_loader.dataset]
    num_classes = len(torch.tensor(labels).unique())

    for trial in range(num_trials):
        if rank == 0:
            print(f"\n---- Starting Trial {trial} ----")
            print(f"- Initializing model...")
        model = ComplexResNet(arch, num_classes=num_classes, activation_function=activation)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        if rank == 0:
            print(f"- Initializing Trainer...")
        trainer = Trainer(model, train_loader, test_loader, optimizer, save_every, trial)
        trainer.train(total_epochs)

    print(f"- Rank {rank} training complete.")
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
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials to run the experiment for.')
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    rank = int(os.environ["SLURM_PROCID"])

    main(rank, args.save_every, args.epochs, args.dataset, args.batch_size, args.architecture, args.activation, args.trials)
