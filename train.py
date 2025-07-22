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
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        print("- Configuring DDP...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

def main(rank: int, save_every: int, total_epochs: int, dataset_name: str, batch_size: int, model_type:str, arch: str, activation: str, num_trials: int, split: list[float]):
    ddp_setup()
    polarization = None
    dataset = None

    if 'S1SLC_CVDL' in dataset_name and len(dataset_name.split('_')) > 2:
        dataset = "_".join(dataset_name.split('_')[0:2])
        polarization = dataset_name.split('_')[-1]

    dataset_name = f"{dataset}_{polarization}" if dataset is not None and polarization is not None else dataset_name

    base_lr = 0.1
    lr = base_lr

    if rank == 0:
        print(f"- Starting Train Loop on Rank {rank} with {torch.cuda.device_count()} GPUs in DDP\n")
        print(f"- Loading Dataset {dataset_name.upper()}...")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=(dataset if dataset is not None else dataset_name), 
        polarization=polarization, 
        batch_size=batch_size, 
        model_type=model_type,
        split=split
    )
    
    num_classes = train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.classes
    num_channels = train_loader.dataset.dataset.channels if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.channels


    for trial in range(num_trials):
        if rank == 0:
            print(f"\n---- Starting Trial {trial} ----")
            print(f"- Initializing model...")
        
        if model_type == 'complex':
            model = ComplexResNet(arch, input_channels=num_channels, num_classes=num_classes, activation_function=activation)
        elif model_type == 'real':
            model = RealResNet(arch, input_channels=num_channels, num_classes=num_classes)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        if rank == 0:
            print(f"- Initializing Trainer...")
        trainer = Trainer(model, dataset_name, train_loader, val_loader, test_loader, optimizer, save_every, trial, polarization, scheduler=scheduler)
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
    parser.add_argument('--save_every', type=int, default=20, help='How often to save a snapshot')
    parser.add_argument('--dataset', type=str, default='S1SLC_CVDL_HH', help='Dataset to use for trainng.')
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials to run the experiment for.')
    parser.add_argument('--model-type', type=str, default='complex', choices=['complex', 'real'])
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.1, 0.1], help='Train, validation, and test split ratios.')
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    if sum(args.split) != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0. Got: {sum(args.split)}")

    rank = int(os.environ["LOCAL_RANK"])
    torch.autograd.set_detect_anomaly(True)

    main(rank, args.save_every, args.epochs, args.dataset, args.batch_size, args.model_type, args.architecture, args.activation, args.trials, args.split)
