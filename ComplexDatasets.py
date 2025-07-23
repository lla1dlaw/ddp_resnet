import os
from math import fsum
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Iterable, Sequence
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
import torchvision.transforms as transforms
import torch.distributed as dist
from tqdm import tqdm

# --- Memory-Efficient Dataset Class ---
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
        if not torch.is_complex(tensor):
            raise TypeError(f"Input tensor should be a complex tensor. Got {tensor.dtype}.")
        return tensor.sub(self.mean).div(self.std)

def _calculate_stats_parallel(dataset: S1SLC_CVDL_Dataset, num_samples_for_stats: int, batch_size: int = 1024):
    """
    Calculates mean and std in parallel across all DDP ranks.
    Only rank 0 will display a progress bar to keep the terminal clean.
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    num_channels = dataset.channels
    
    samples_per_rank = int(np.ceil(num_samples_for_stats / world_size))
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, num_samples_for_stats)
    
    local_samples_to_process = end_idx - start_idx
    
    local_sum = np.zeros(num_channels, dtype=np.float64)
    
    # --- MODIFIED: TQDM progress bar only shows on rank 0 ---
    with tqdm(total=local_samples_to_process, desc="Calculating Mean (Rank 0)", unit="sample", disable=(rank != 0)) as pbar:
        global_samples_processed = 0
        for info in dataset.file_info:
            file_start_global = global_samples_processed
            file_end_global = file_start_global + info['size']
            
            overlap_start = max(start_idx, file_start_global)
            overlap_end = min(end_idx, file_end_global)

            if overlap_start < overlap_end:
                local_start = overlap_start - file_start_global
                local_end = overlap_end - file_start_global
                
                hh_mmap = np.load(info['hh'], mmap_mode='r')
                hv_mmap = np.load(info['hv'], mmap_mode='r')

                for i in range(local_start, local_end, batch_size):
                    batch_end_local = min(i + batch_size, local_end)
                    
                    if dataset.polarization == 'HH': batch_complex = hh_mmap[i:batch_end_local]
                    elif dataset.polarization == 'HV': batch_complex = hv_mmap[i:batch_end_local]
                    else: batch_complex = np.stack([hh_mmap[i:batch_end_local], hv_mmap[i:batch_end_local]], axis=1)

                    if dataset.dtype == 'real': batch = np.concatenate([batch_complex.real, batch_complex.imag], axis=1)
                    else: batch = batch_complex
                    
                    local_sum += np.sum(batch, axis=(0, 2, 3))
                    if rank == 0: pbar.update(batch_end_local - i)

            global_samples_processed += info['size']
            if global_samples_processed >= end_idx:
                break

    global_sum_tensor = torch.from_numpy(local_sum).to(f"cuda:{rank}")
    if is_distributed:
        dist.all_reduce(global_sum_tensor, op=dist.ReduceOp.SUM)
    
    count = num_samples_for_stats * 100 * 100
    mean = (global_sum_tensor / count).cpu().numpy()
    mean_reshaped = mean.reshape(1, num_channels, 1, 1)

    local_sq_diff = np.zeros(num_channels, dtype=np.float64)
    
    # --- MODIFIED: TQDM progress bar only shows on rank 0 ---
    with tqdm(total=local_samples_to_process, desc="Calculating Std Dev (Rank 0)", unit="sample", disable=(rank != 0)) as pbar:
        global_samples_processed = 0
        for info in dataset.file_info:
            file_start_global = global_samples_processed
            file_end_global = file_start_global + info['size']

            overlap_start = max(start_idx, file_start_global)
            overlap_end = min(end_idx, file_end_global)

            if overlap_start < overlap_end:
                local_start = overlap_start - file_start_global
                local_end = overlap_end - file_start_global

                hh_mmap = np.load(info['hh'], mmap_mode='r')
                hv_mmap = np.load(info['hv'], mmap_mode='r')

                for i in range(local_start, local_end, batch_size):
                    batch_end_local = min(i + batch_size, local_end)
                    
                    if dataset.polarization == 'HH': batch_complex = hh_mmap[i:batch_end_local]
                    elif dataset.polarization == 'HV': batch_complex = hv_mmap[i:batch_end_local]
                    else: batch_complex = np.stack([hh_mmap[i:batch_end_local], hv_mmap[i:batch_end_local]], axis=1)
                        
                    if dataset.dtype == 'real': batch = np.concatenate([batch_complex.real, batch_complex.imag], axis=1)
                    else: batch = batch_complex

                    local_sq_diff += np.sum((batch - mean_reshaped) ** 2, axis=(0, 2, 3))
                    if rank == 0: pbar.update(batch_end_local - i)
            
            global_samples_processed += info['size']
            if global_samples_processed >= end_idx:
                break

    global_sq_diff_tensor = torch.from_numpy(local_sq_diff).to(f"cuda:{rank}")
    if is_distributed:
        dist.all_reduce(global_sq_diff_tensor, op=dist.ReduceOp.SUM)
        
    variance = global_sq_diff_tensor.cpu().numpy() / count
    std = np.sqrt(variance)
    
    return mean, std

def S1SLC_CVDL(
        root: Union[str, Path],
        polarization: str,
        dtype: str,
        split: Optional[Iterable] = None,
        **kwargs
) -> list[Subset]:
    
    _validate_args(root, "S1SLC_CVDL", polarization, split)
    
    full_dataset = S1SLC_CVDL_Dataset(root, "S1SLC_CVDL", polarization, dtype)
    
    train_size = int(split[0] * len(full_dataset))
    
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        print(f"Calculating normalization stats in parallel across {dist.get_world_size()} ranks...")
    
    mean, std = _calculate_stats_parallel(full_dataset, train_size)
    
    full_dataset.set_normalization(mean, std)
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = '0'
        
    print("--- Running Direct Test of ComplexDatasets.py ---")
    trainset, valset, testset = S1SLC_CVDL(root='./data', polarization=None, dtype='real', split=[0.8, 0.1, 0.1])
    
    print(f"\nTraining set length: {len(trainset)}")
    print(f"Validation set length: {len(valset)}")
    print(f"Test set length: {len(testset)}")

    sample_data, sample_label = trainset[0]
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data type: {sample_data.dtype}")
    print(f"Sample label: {sample_label}")
