import pretty_errors
import os
from math import fsum
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Iterable
from dotenv import load_dotenv
from s3torchconnector import S3MapDataset
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
import torchvision.transforms as transforms
from ProgressFile import ProgressFile
import contextlib


class CustomDataset(Dataset):
    def __init__(
            self,
            tensors: Iterable[np.ndarray | torch.Tensor],
            num_classes: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.tensors = [torch.as_tensor(tensor) for tensor in tensors]
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = num_classes
        self.channels = tensors[0].shape[1]

    def __getitem__(self, index):
        data = self.tensors[0][index]
        target = self.tensors[1][index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return self.tensors[0].size(0)


class ComplexNormalize:
    def __init__(self, mean: complex, std: float):
        self.mean = torch.as_tensor(mean, dtype=torch.complex64)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        if self.mean.ndim == 1: self.mean = self.mean.view(-1, 1, 1)
        if self.std.ndim == 1: self.std = self.std.view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(tensor):
            raise TypeError(f"Input tensor should be a complex tensor. Got {tensor.dtype}.")
        return tensor.sub_(self.mean).div_(self.std)

def S1SLC_CVDL(
        root: Union[str, Path],
        polarization: str,
        dtype: str,
        split: Optional[Iterable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_s3: bool = False,
) -> list[Subset]:
    if use_s3:
        raise NotImplementedError("S3 streaming is not configured in this version.")
    else: 
        base_dir = "mini_S1SLC_CVDL"
        return _load_complex_dataset(
            root_dir=root, base_dir=base_dir, dtype=dtype,
            training_split=split, transform=transform,
            target_transform=target_transform, polarization=polarization,
        )

def _load_complex_dataset(
    root_dir: str,
    base_dir:str,
    dtype: str,
    transform: Optional[Callable],
    target_transform: Optional[Callable],
    polarization: Optional[str] = None,
    training_split: Optional[Iterable] = [0.8, 0.1, 0.1],
) -> list[Subset]:
    rank = int(os.environ["LOCAL_RANK"])
    validate_args(root_dir, base_dir, polarization, training_split)

    HH_data, HV_data, label_data = [], [], []
    path = os.path.join(root_dir, base_dir)
    for dir in os.listdir(path):
        data_dir = os.path.join(path, dir)
        HH_data.extend(_load_np_from_file(os.path.join(data_dir, 'HH_Complex_Patches.npy'), rank))
        HV_data.extend(_load_np_from_file(os.path.join(data_dir, 'HV_Complex_Patches.npy'), rank))
        label_data.extend(_load_np_from_file(os.path.join(data_dir, 'Labels.npy'), rank))

    HH_array = np.expand_dims(np.array(HH_data), axis=1)
    HV_array = np.expand_dims(np.array(HV_data), axis=1)
    if polarization == 'HH':
        inputs = HH_array
    elif polarization == 'HV':
        inputs = HV_array
    else:
        inputs = np.concatenate((HH_array, HV_array), axis=1)
    
    labels = np.array(label_data).squeeze().astype(np.int64) - 1
    shuffle_arrays(inputs, labels)

    num_samples = len(inputs)
    train_size = int(training_split[0] * num_samples)
    train_inputs_for_stats = inputs[:train_size]

    if dtype == 'real':
        train_stats_data = np.concatenate((train_inputs_for_stats.real, train_inputs_for_stats.imag), axis=1)
        mean, std = np.mean(train_stats_data, axis=(0, 2, 3)), np.std(train_stats_data, axis=(0, 2, 3))
        final_transform = transforms.Normalize(mean, std)
        inputs = np.concatenate((inputs.real, inputs.imag), axis=1)
    elif dtype == 'complex':
        mean, std = np.mean(train_inputs_for_stats, axis=(0, 2, 3)), np.std(train_inputs_for_stats, axis=(0, 2, 3))
        final_transform = ComplexNormalize(mean, std)

    full_dataset = CustomDataset(
        (inputs, labels),
        num_classes=7,
        transform=final_transform
    )

    return random_split(full_dataset, training_split, generator=torch.Generator().manual_seed(42))

def shuffle_arrays(*arrays):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(2**32)
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

def _load_np_from_file(path: str, rank: int) -> np.array:
    progress_context = ProgressFile(path, "rb", desc=f'reading {path}') if rank == 0 else contextlib.nullcontext()
    with progress_context as f:
        array = np.load(f if f is not None else path)
        if array.dtype == np.complex128:
            array = array.astype(np.complex64)
    return array

def validate_args(root_dir: str, base_dir:str, polarization: Optional[str], training_split: Iterable[float]) -> None:
    path = os.path.join(root_dir, base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find directory: {path}")
    if polarization not in ['HH', 'HV', None]:
        raise ValueError(f"Unknown argument for polarization {polarization}")
    if training_split is not None and not np.isclose(fsum(training_split), 1.0):
        raise ValueError(f"Values in training_split must sum to 1. Got: {fsum(training_split)}")
