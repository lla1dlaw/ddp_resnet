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

# --- NEW: Memory-Efficient Dataset Class ---
class S1SLC_CVDL_Dataset(Dataset):
    """
    A memory-efficient dataset for the S1SLC_CVDL data.
    This class reads data samples from .npy files on-the-fly instead of
    loading the entire dataset into RAM. It uses numpy's memory-mapping
    for efficient file access.
    """
    def __init__(self, root_dir: str, base_dir: str, polarization: Optional[str], dtype: str):
        super().__init__()
        self.root_dir = root_dir
        self.base_dir = base_dir
        self.polarization = polarization
        self.dtype = dtype
        self.transform = None # Normalization is set later
        self.classes = ['AG', 'FR', 'HD', 'HR', 'LD', 'IR', 'WR']
        self.num_classes = len(self.classes)

        self._file_info = []
        self._cumulative_sizes = [0]
        
        path = os.path.join(self.root_dir, self.base_dir)
        city_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

        for city in city_dirs:
            data_dir = os.path.join(path, city)
            hh_path = os.path.join(data_dir, 'HH_Complex_Patches.npy')
            hv_path = os.path.join(data_dir, 'HV_Complex_Patches.npy')
            labels_path = os.path.join(data_dir, 'Labels.npy')

            if not all(os.path.exists(p) for p in [hh_path, hv_path, labels_path]):
                continue

            with np.load(labels_path) as f:
                num_samples = len(f)
            
            self._file_info.append({
                'hh': hh_path,
                'hv': hv_path,
                'labels': labels_path,
                'size': num_samples
            })
            self._cumulative_sizes.append(self._cumulative_sizes[-1] + num_samples)

        if self.polarization is None:
            self.channels = 4 if self.dtype == 'real' else 2
        else:
            self.channels = 2 if self.dtype == 'real' else 1

    def __len__(self):
        return self._cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Find which city's file this index belongs to
        city_idx = next(i for i, size in enumerate(self._cumulative_sizes) if size > index) - 1
        local_index = index - self._cumulative_sizes[city_idx]
        
        info = self._file_info[city_idx]

        # Use memory-mapping to avoid loading the whole file
        hh_data = np.load(info['hh'], mmap_mode='r')[local_index]
        hv_data = np.load(info['hv'], mmap_mode='r')[local_index]
        label = np.load(info['labels'], mmap_mode='r')[local_index]

        # Combine polarizations
        if self.polarization == 'HH':
            sample_complex = np.expand_dims(hh_data, axis=0)
        elif self.polarization == 'HV':
            sample_complex = np.expand_dims(hv_data, axis=0)
        else: # Both HH and HV
            sample_complex = np.stack([hh_data, hv_data], axis=0)

        # Handle real vs complex dtype
        if self.dtype == 'real':
            sample = np.concatenate([sample_complex.real, sample_complex.imag], axis=0).astype(np.float32)
        else:
            sample = sample_complex.astype(np.complex64)
        
        # Convert to tensor and apply transforms
        sample_tensor = torch.from_numpy(sample)
        if self.transform:
            sample_tensor = self.transform(sample_tensor)
            
        # Squeeze label and correct for 1-based indexing
        target = torch.tensor(label.squeeze() - 1, dtype=torch.long)
        
        return sample_tensor, target

    def set_normalization(self, mean, std):
        if self.dtype == 'real':
            self.transform = transforms.Normalize(mean, std)
        else:
            self.transform = ComplexNormalize(mean, std)

class ComplexNormalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.from_numpy(mean).cfloat()
        self.std = torch.from_numpy(std).cfloat().abs() # Std dev is real
        if self.mean.ndim == 1: self.mean = self.mean.view(-1, 1, 1)
        if self.std.ndim == 1: self.std = self.std.view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(tensor):
            raise TypeError(f"Input tensor should be a complex tensor. Got {tensor.dtype}.")
        return tensor.sub(self.mean).div(self.std)

def _get_iterative_stats_from_dataset(dataset: S1SLC_CVDL_Dataset, num_samples_for_stats: int, batch_size: int = 256):
    """Calculates mean and std from the lazy-loading dataset."""
    num_batches = (num_samples_for_stats + batch_size - 1) // batch_size
    
    # First pass: Calculate mean
    sum_data = 0
    # Important: Iterate only over the training split portion for stats
    for i in range(0, num_samples_for_stats, batch_size):
        batch_end = min(i + batch_size, num_samples_for_stats)
        # Create a batch by calling __getitem__ multiple times
        batch = torch.stack([dataset[j][0] for j in range(i, batch_end)])
        sum_data += torch.sum(batch, dim=(0, 2, 3), keepdim=True)

    count = num_samples_for_stats * dataset[0][0].shape[1] * dataset[0][0].shape[2]
    mean = sum_data / count
    
    # Second pass: Calculate variance
    sum_sq_diff = 0
    for i in range(0, num_samples_for_stats, batch_size):
        batch_end = min(i + batch_size, num_samples_for_stats)
        batch = torch.stack([dataset[j][0] for j in range(i, batch_end)])
        sum_sq_diff += torch.sum((batch - mean) ** 2, dim=(0, 2, 3), keepdim=True)
        
    variance = sum_sq_diff / count
    std = torch.sqrt(variance)
    
    return mean.squeeze().numpy(), std.squeeze().numpy()

def S1SLC_CVDL(
        root: Union[str, Path],
        polarization: str,
        dtype: str,
        split: Optional[Iterable] = None,
        **kwargs # other args are ignored but needed for compatibility
) -> list[Subset]:
    
    _validate_args(root, "S1SLC_CVDL", polarization, split)
    
    # 1. Create the full dataset object (loads no data yet)
    full_dataset = S1SLC_CVDL_Dataset(root, "S1SLC_CVDL", polarization, dtype)
    
    # 2. Calculate stats ONLY on the training portion
    train_size = int(split[0] * len(full_dataset))
    print(f"Calculating normalization stats from {train_size} training samples...")
    mean, std = _get_iterative_stats_from_dataset(full_dataset, train_size)
    
    # 3. Set the calculated normalization transform on the dataset
    full_dataset.set_normalization(mean, std)
    
    # 4. Split the dataset into train, validation, and test subsets
    # Use a generator for reproducible splits
    return random_split(full_dataset, split, generator=torch.Generator().manual_seed(42))

def _validate_args(root_dir: str, base_dir:str, polarization: Optional[str], training_split: Iterable[float]) -> None:
    path = os.path.join(root_dir, base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find directory: {path}")
    if polarization not in ['HH', 'HV', None]:
        raise ValueError(f"Unknown argument for polarization {polarization}")
    if training_split is not None and not np.isclose(fsum(training_split), 1.0):
        raise ValueError(f"Values in training_split must sum to 1. Got: {fsum(training_split)}")

if __name__ == "__main__":
    # Mock environment variable for direct script execution
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = '0'
        
    print("--- Running Direct Test of ComplexDatasets.py ---")
    trainset, valset, testset = S1SLC_CVDL(root='./data', polarization=None, dtype='real', split=[0.8, 0.1, 0.1])
    
    print(f"\nTraining set length: {len(trainset)}")
    print(f"Validation set length: {len(valset)}")
    print(f"Test set length: {len(testset)}")

    # Verify a sample can be loaded
    sample_data, sample_label = trainset[0]
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data type: {sample_data.dtype}")
    print(f"Sample label: {sample_label}")
