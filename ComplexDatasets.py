import pretty_errors
import os
from math import fsum
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Iterable, Sequence
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
import torchvision.transforms as transforms
from ProgressFile import ProgressFile
import contextlib


class CustomDataset(Dataset):
    def __init__(
            self,
            tensors: Sequence[np.ndarray | Tensor],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.classes = ['AG', 'FR', 'HD', 'HR', 'LD', 'IR', 'WR']
        self.num_classes = len(self.classes)
        
        self.tensors = [torch.as_tensor(tensor) for tensor in tensors]
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors)
        self.transform = transform
        self.target_transform = target_transform
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
        base_dir = "S1SLC_CVDL"
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
    training_split: Sequence[float]=[0.8, 0.1, 0.1],
) -> list[Subset]:
    rank = int(os.environ["LOCAL_RANK"])
    _validate_args(root_dir, base_dir, polarization, training_split)

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
        mean, std = _get_iterative_stats(train_stats_data)
        final_transform = transforms.Normalize(mean, std)
        inputs = np.concatenate((inputs.real, inputs.imag), axis=1)
    elif dtype == 'complex':
        mean, std = _get_iterative_stats(train_inputs_for_stats)
        final_transform = ComplexNormalize(mean, std)

    full_dataset = CustomDataset(
        (inputs, labels),
        transform=final_transform
    )

    return random_split(full_dataset, training_split)

def _get_iterative_stats(data: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates mean and std of a large array iteratively to avoid memory errors.
    """
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    # First pass: Calculate mean
    sum_data = 0
    for i in range(num_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        sum_data += np.sum(batch, axis=(0, 2, 3), keepdims=True)

    # Shape of data is (N, C, H, W). We sum over N, H, W, leaving C.
    # Total elements per channel is N * H * W
    count = num_samples * data.shape[2] * data.shape[3]
    mean = sum_data / count

    # Second pass: Calculate variance and std
    sum_sq_diff = 0
    for i in range(num_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        sum_sq_diff += np.sum((batch - mean) ** 2, axis=(0, 2, 3), keepdims=True)

    variance = sum_sq_diff / count
    std = np.sqrt(variance)

    # Squeeze out the H and W dimensions (which are 1) to get shape (C,)
    return mean.squeeze(), std.squeeze()

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

def _validate_args(root_dir: str, base_dir:str, polarization: Optional[str], training_split: Iterable[float]) -> None:
    path = os.path.join(root_dir, base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find directory: {path}")
    if polarization not in ['HH', 'HV', None]:
        raise ValueError(f"Unknown argument for polarization {polarization}")
    if training_split is not None and not np.isclose(fsum(training_split), 1.0):
        raise ValueError(f"Values in training_split must sum to 1. Got: {fsum(training_split)}")


if __name__ == "__main__":
    os.environ["LOCAL_RANK"] = '0'
    trainset, valset, testset = S1SLC_CVDL(root='./data', polarization=None, dtype='real', split=[0.8, 0.1, 0.1], transform=None)
    print(f"Training set length: {len(trainset)}")
    print(f"Validation set length: {len(valset)}")
    print(f"Test set length: {len(testset)}")
