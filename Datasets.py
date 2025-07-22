import os
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from ComplexDatasets import S1SLC_CVDL
import torch

dataset_map = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
    'S1SLC_CVDL': S1SLC_CVDL,
}

def get_dataset(dataset_name: str, polarization, model_type: str, split: list[float]):
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        print(f"Begining Load Process for {dataset_name}")
    
    if dataset_name in ['cifar10', 'cifar100', 'mnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        full_trainset = dataset_map[dataset_name.lower()](root='./data', train=True, download=True, transform=transform)
        testset = dataset_map[dataset_name.lower()](root='./data', train=False, download=True, transform=transform)
        
        train_size = int(0.9 * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])

    elif dataset_name == "S1SLC_CVDL":
        transform = None
        trainset, valset, testset = dataset_map['S1SLC_CVDL'](root='./data', polarization=polarization, dtype=model_type, split=split, transform=transform)
    
    if rank == 0:
        print(f"{dataset_name.upper()} datasets loaded successfully.")
    return trainset, valset, testset

def get_dataloaders(dataset_name: str, polarization, batch_size: int, model_type: str, split: list[float]) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = get_dataset(dataset_name, polarization, model_type, split)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(train_set), num_workers=4,
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(val_set), num_workers=4,
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(test_set), num_workers=4,
    )

    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        train_image, _ = next(iter(train_loader))
        print(f"Sample Train Batch Shape: {train_image.size()}")

    return train_loader, val_loader, test_loader
