import torch
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

dataset_map: dict[str, Dataset] = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
}

def get_datasets(dataset_name: str) -> tuple[Dataset, Dataset]:
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
    print(f"{dataset_name.upper()} datasets loaded successfully.")
    return trainset, testset



def get_dataloaders(dataset_name: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = get_datasets(dataset_name)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_set) 
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_set) 
    )

    return train_loader, test_loader
