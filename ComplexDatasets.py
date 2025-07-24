import os
from math import fsum
import numpy as np
from pathlib import Path
from typing import Union, Optional, Iterable
import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


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

        self.file_info = []
        self.cumulative_sizes = [0]
        
        path = os.path.join(self.root_dir, self.base_dir)
        city_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

        for city in city_dirs:
            data_dir = os.path.join(path, city)
            hh_path = os.path.join(data_dir, 'HH_Complex_Patches.npy')
            hv_path = os.path.join(data_dir, 'HV_Complex_Patches.npy')
            labels_path = os.path.join(data_dir, 'Labels.npy')

            if not all(os.path.exists(p) for p in [hh_path, hv_path, labels_path]):
                continue

            labels_array = np.load(labels_path)
            num_samples = len(labels_array)
            
            self.file_info.append({
                'hh': hh_path,
                'hv': hv_path,
                'labels': labels_path,
                'size': num_samples
            })
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_samples)

        if self.polarization is None:
            # For real dtype, complex data is split into real and imag parts, doubling channels
            self.channels = 4 if self.dtype == 'real' else 2
        else:
            self.channels = 2 if self.dtype == 'real' else 1

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        city_idx = next(i for i, size in enumerate(self.cumulative_sizes) if size > index) - 1
        local_index = index - self.cumulative_sizes[city_idx]
        
        info = self.file_info[city_idx]

        hh_data = np.load(info['hh'], mmap_mode='r')[local_index]
        hv_data = np.load(info['hv'], mmap_mode='r')[local_index]
        label = np.load(info['labels'], mmap_mode='r')[local_index]

        if self.polarization == 'HH':
            sample_complex = np.expand_dims(hh_data, axis=0)
        elif self.polarization == 'HV':
            sample_complex = np.expand_dims(hv_data, axis=0)
        else:
            sample_complex = np.stack([hh_data, hv_data], axis=0)

        if self.dtype == 'real':
            sample = np.concatenate([sample_complex.real, sample_complex.imag], axis=0).astype(np.float32)
        else:
            sample = sample_complex.astype(np.complex64)
        
        sample_tensor = torch.from_numpy(sample)
        if self.transform:
            sample_tensor = self.transform(sample_tensor)
            
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
        self.std = torch.from_numpy(std).cfloat().abs()
        if self.mean.ndim == 1: self.mean = self.mean.view(-1, 1, 1)
        if self.std.ndim == 1: self.std = self.std.view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(tensor) and self.mean.is_complex():
            raise TypeError(f"Input tensor must be complex. Got {tensor.dtype}.")
        return tensor.sub(self.mean).div(self.std)

# --- CHANGE: Replaced `_calculate_stats_parallel` with a robust, correct implementation ---
def _calculate_distributed_stats(data_loader: DataLoader, num_channels: int, dtype: str, rank: int):
    """
    Performs a true parallel calculation of dataset statistics across all DDP ranks.
    This version correctly handles complex data types and uses the proper variance formula.
    """
    device = torch.device(f"cuda:{rank}")
    
    # --- Pass 1: Calculate Global Mean ---
    accumulator_dtype = torch.complex128 if dtype == 'complex' else torch.float64
    local_sum = torch.zeros(num_channels, dtype=accumulator_dtype, device=device)
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Calculating Mean (Pass 1)", unit="batch", disable=(rank != 0))
    for samples, _ in progress_bar:
        samples = samples.to(device)
        local_sum += torch.sum(samples, dim=(0, 2, 3))
        total_samples += len(samples)

    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    pixel_count = total_samples_tensor.item() * samples.size(2) * samples.size(3)
    mean = local_sum / pixel_count

    local_sq_diff = torch.zeros(num_channels, dtype=torch.float64, device=device)
    mean_reshaped = mean.view(1, num_channels, 1, 1)

    progress_bar = tqdm(data_loader, desc="Calculating Std Dev (Pass 2)", unit="batch", disable=(rank != 0))
    for samples, _ in progress_bar:
        samples = samples.to(device)
        sq_diff = (samples - mean_reshaped).abs() ** 2
        local_sq_diff += torch.sum(sq_diff, dim=(0, 2, 3))

    dist.all_reduce(local_sq_diff, op=dist.ReduceOp.SUM)

    variance = local_sq_diff / pixel_count
    std = torch.sqrt(variance)

    return mean.cpu().numpy(), std.cpu().numpy()


def S1SLC_CVDL(
        root: Union[str, Path],
        polarization: str,
        dtype: str,
        split: Optional[Iterable] = None,
        **kwargs
) -> list[Subset]:
    
    _validate_args(root, "S1SLC_CVDL", polarization, split)

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    full_dataset = S1SLC_CVDL_Dataset(root, "S1SLC_CVDL", polarization, dtype)
    
    # Split the dataset indices before calculating stats
    train_set, val_set, test_set = random_split(full_dataset, split, generator=torch.Generator().manual_seed(42))

    # Create a temporary DataLoader for the training set to calculate stats
    stats_loader = DataLoader(
        train_set,
        batch_size=kwargs.get('stats_batch_size', 256),
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False)
    )
    
    if rank == 0:
        print(f"Calculating normalization stats in parallel across {world_size} ranks...")
    
    # All ranks participate in the calculation
    mean_val, std_val = _calculate_distributed_stats(stats_loader, full_dataset.channels, dtype, rank)

    # Adapt stats for the 'real' model type if needed
    if dtype == 'real':
        # For real networks, mean is split and std is duplicated for real/imag channels
        mean_for_norm = np.concatenate([mean_val.real, mean_val.imag])
        std_for_norm = np.concatenate([std_val, std_val])
    else:
        mean_for_norm = mean_val
        std_for_norm = std_val

    # Set the normalization transform on the root dataset object
    full_dataset.set_normalization(mean_for_norm, std_for_norm)
    
    if rank == 0:
        print("Normalization stats applied.")
    
    if is_distributed:
        dist.barrier()

    return [train_set, val_set, test_set]


def _validate_args(root_dir: str, base_dir:str, polarization: Optional[str], training_split: Iterable[float]) -> None:
    path = os.path.join(root_dir, base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find directory: {path}")
    if polarization not in ['HH', 'HV', None]:
        raise ValueError(f"Unknown argument for polarization {polarization}")
    if training_split is not None and not np.isclose(fsum(training_split), 1.0):
        raise ValueError(f"Values in training_split must sum to 1. Got: {fsum(training_split)}")
if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = '0'
        
    print("--- Running Direct Test of ComplexDatasets.py for complex dtype ---")
    trainset, valset, testset = S1SLC_CVDL(root='./data', polarization=None, dtype='complex', split=[0.8, 0.1, 0.1])
    
    print(f"\nTraining set length: {len(trainset)}")
    print(f"Validation set length: {len(valset)}")
    print(f"Test set length: {len(testset)}")

    sample_data, sample_label = trainset[0]
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data type: {sample_data.dtype}")
    print(f"Sample label: {sample_label}")

    print("--- Running Direct Test of ComplexDatasets.py for real dtype ---")
    trainset, valset, testset = S1SLC_CVDL(root='./data', polarization=None, dtype='real', split=[0.8, 0.1, 0.1])
    
    print(f"\nTraining set length: {len(trainset)}")
    print(f"Validation set length: {len(valset)}")
    print(f"Test set length: {len(testset)}")

    sample_data, sample_label = trainset[0]
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data type: {sample_data.dtype}")
    print(f"Sample label: {sample_label}")
