import os
from torch import polar
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from ComplexDatasets import S1SLC_CVDL
from datetime import datetime

dataset_map = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
    'S1SLC_CVDL': S1SLC_CVDL,
}

def get_dataset(dataset_name: str, polarization):
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        print(f"Begining Load Process for {dataset_name}")
    if dataset_name in ['cifar10', 'cifar100', 'mnist']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = dataset_map[dataset_name.lower()](root='./data', train=True, download=True, transform=transform_train)
        testset = dataset_map[dataset_name.lower()](root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == "S1SLC_CVDL":
        transform = None # define complex transform here
        trainset, testset = dataset_map['S1SLC_CVDL'](root='./data', polarization=polarization, dtype='real', split=[0.8, 0.2], transform=transform)
    if rank == 0:
        print(f"{dataset_name.upper()} datasets loaded successfully.")
    return trainset, testset


def get_dataloaders(dataset_name: str, polarization, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = get_dataset(dataset_name, polarization)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_set), 
        num_workers=4,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_set),
        num_workers=4,
    )

    return train_loader, test_loader

