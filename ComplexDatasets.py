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
from ProgressFile import ProgressFile
import contextlib


class CustomDataset(Dataset):
    def __init__(
            self,
            tensors: Iterable[np.ndarray | torch.Tensor],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        # Ensure all tensors have the same first dimension (number of samples)
        self.tensors = [torch.as_tensor(tensor) for tensor in tensors]
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.tensors[0][index]
        target = self.tensors[1][index] # Assuming the second tensor is the target
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return self.tensors[0].size(0)


def S1SLC_CVDL( # Call this method only.
        root: Union[str, Path],
        polarization: str,
        dtype: str,
        split: Optional[Iterable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_s3: bool = False, 
) -> Union[S3MapDataset, CustomDataset, list[Subset[tuple[Tensor,...]]]]:

    if use_s3:
        return _get_S3_stream(transform)
    else: 
        base_dir = "mini_S1SLC_CVDL"
        return _load_complex_dataset(
            root_dir=root,
            base_dir=base_dir,
            dtype=dtype,
            training_split=split,
            transform=transform,
            target_transform=target_transform,
            polarization=polarization,
        )

def _load_complex_dataset(
    root_dir: str,
    base_dir:str,
    dtype: str,
    transform: Optional[Callable],
    target_transform: Optional[Callable],
    polarization: Optional[str] = None,
    training_split: Optional[Iterable] = [0.8, 0.2],
) -> Union[CustomDataset, list[Subset[tuple[Tensor,...]]]]:
    rank = int(os.environ["LOCAL_RANK"])
    validate_args(root_dir, base_dir, polarization, training_split)

    HH_data = []
    HV_data = []
    label_data = []

    path = os.path.join(root_dir, base_dir)
    for dir in os.listdir(path):
        data_dir = os.path.join(path, dir)

        HH_path = os.path.join(data_dir, 'HH_Complex_Patches.npy')
        HV_path = os.path.join(data_dir, 'HV_Complex_Patches.npy')
        labels_path = os.path.join(data_dir, 'Labels.npy')
        
        if rank == 0:
            print(f"\nLoading S1SLC_CVDL {dir}")
        HH_data.extend(_load_np_from_file(HH_path, rank))
        HV_data.extend(_load_np_from_file(HV_path, rank))
        label_data.extend(_load_np_from_file(labels_path, rank))

    HH_array = np.array(HH_data)
    HV_array = np.array(HV_data)

    if polarization == 'HH':
        inputs = np.expand_dims(HH_array, axis=1)
    elif polarization == 'HV':
        inputs = np.expand_dims(HV_array, axis=1)
    elif polarization == 'both': # treats each set of data as a separate channel 
        inputs = np.stack((HH_array, HV_array), axis=1) 

    if dtype == 'real': # stacks real and imaginary components together and doubles channels as a result
        real = inputs.real
        imag = inputs.imag
        inputs = np.concatenate((real, imag), axis=1)

    labels = np.array(label_data).squeeze().astype(np.int64) - 1
    dataset = CustomDataset(
        (inputs, labels),
        transform=transform,
        target_transform=target_transform,
    )

    if training_split is None:
        return dataset

    dataset_sections = random_split(dataset, training_split)
    return dataset_sections


def _load_np_from_file(path: str, rank: int) -> np.array:
    """ Helper function to load a saved numpy array from a .npy file """

    progress_context = ProgressFile(path, "rb", desc=f'reading {path}') if rank == 0 else contextlib.nullcontext()

    with progress_context as f:
        if f is not None:
            array = np.load(f)
        else:
            array = np.load(path)
        if array.dtype == np.complex128:
            array = array.astype(np.complex64)# decrease size for memory savings
        f.close()
    return array


def save_arrays(path:str = "./data/mini_S1SLC_CVDL/", **paths_and_arrays):
    os.makedirs(path, exist_ok=True)
    for name, array in paths_and_arrays.items():
        filename = f'{name}.npy'
        np.save(os.path.join(path, filename), array, allow_pickle=True)



def balance_dataset_multi(*image_arrays, labels, n_samples_per_class, random_sample=True):
    """
    Balances a dataset by selecting samples based on shared labels.

    This function works for one or more image arrays that are all aligned
    with a single labels array.

    Args:
        *image_arrays (np.ndarray): A variable number of data arrays (e.g., images).
                                    All must have the same length as `labels`.
        labels (np.ndarray): A 1D array of class labels. Assumed to be integers.
        n_samples_per_class (int): The number of samples to store from each class. 
        random_sample (bool): If True, select a random sample. If False, select
                              the first n samples. Defaults to True.

    Returns:
        tuple: A tuple containing the balanced version of each input array,
               followed by the balanced labels array at the end.
    """
    all_selected_indices = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]

        if len(class_indices) >= n_samples_per_class:
            if random_sample:
                selected_indices = np.random.choice(class_indices, n_samples_per_class, replace=False)
            else:
                selected_indices = class_indices[:n_samples_per_class]
        else:
            print(f"⚠️ Warning: Class {label} has only {len(class_indices)} samples. Taking all of them.")
            selected_indices = class_indices
        
        all_selected_indices.extend(selected_indices)

    final_indices = np.array(all_selected_indices, dtype=int)
    print(final_indices[0:5])
    balanced_labels = labels[final_indices]
    balanced_image_arrays = [arr[final_indices] for arr in image_arrays]
    classes, confirmed_samples_per_class = np.unique(balanced_labels, return_counts=True)

    print("Final count for each class:")
    
    for classification, count in zip(classes, confirmed_samples_per_class):
        print(f"\t{classification} - {count}")
    
    return (*balanced_image_arrays, balanced_labels)


def make_mini_dataset(
    root_dir: str = "./data",
    base_dir:str = "S1SLC_CVDL",
    num_samples: int = 10000, # 10,000 samples per class
) -> None:

    HH_data = []
    HV_data = []
    label_data = []

    path = os.path.join(root_dir, base_dir)
    for dir in os.listdir(path):
        data_dir = os.path.join(path, dir)

        HH_path = os.path.join(data_dir, 'HH_Complex_Patches.npy')
        HV_path = os.path.join(data_dir, 'HV_Complex_Patches.npy')
        labels_path = os.path.join(data_dir, 'Labels.npy')

        print(f"\nLoading S1SLC_CVDL {dir}")
        HH_data.extend(_load_np_from_file(HH_path))
        HV_data.extend(_load_np_from_file(HV_path))
        label_data.extend(_load_np_from_file(labels_path))

    print(f"Num HH: {len(HH_data)}")
    print(f"Num HV: {len(HV_data)}")
    print(f"Num Labels: {len(label_data)}")

    HH_array = np.array(HH_data)
    HV_array = np.array(HV_data)
    labels_array = np.array(label_data)


    HH_Complex_Patches, HV_Complex_Patches, Labels = balance_dataset_multi(HH_array, HV_array, labels=labels_array, n_samples_per_class=num_samples)
    paths_and_arrays = {
        Path(HH_path).stem: HH_Complex_Patches,
        Path(HV_path).stem: HV_Complex_Patches,
        Path(labels_path).stem: Labels,
    }
    save_arrays(path="./data/mini_S1SLC_CVDL/", **paths_and_arrays)


def validate_args(
    root_dir: str,
    base_dir:str,
    polarization: Optional[str],
    training_split: Iterable[float]
) -> None:
    path = os.path.join(root_dir, base_dir)
    try:
        os.path.exists(path)
    except FileNotFoundError:
        print(f"Could not find directory: {path}")
    
    if polarization not in ['HH', 'HV'] and polarization is not None:
        raise ValueError(f"Unkonwn argument for polarization {polarization}")

    if training_split is not None and fsum(training_split) != 1.0:
        raise ValueError(f"Values in training_split must sum to 1. Got: {fsum(training_split)}")


def _get_S3_stream(transform: Optional[Callable] = None) -> S3MapDataset:
    URI = "s3://ieee-dataport/open/98396/S1SLC_CVDL.rar"
    REGION = "us-east-1"
    try:
        load_dotenv() # AWS secret ID & Secret key must be stored in a .env file for accessing the SLSLC_CVDL. If such a file does not exist, create one.
        if transform is not None:
            return S3MapDataset.from_prefix(URI, region=REGION, transform=transform) # S3MapDataset streams data from an S3 bucket rather than downloading the full dataset. 
        return S3MapDataset.from_prefix(URI, region=REGION)
    except FileNotFoundError:
        print(".env file not found in project root directory and is required for download.")
        print("Create this file in the project root and place your IEEE-dataport AWS secret ID and secret key inside with the following format:")
        print("\tAWS_ACCESS_KEY_ID=<your ID>\n\tAWS_SECRET_ACCESS_KEY=<your secret key>")
        print("Or remove 'download = True' argument from S1SLC_CVDL call to attempt local loading.")
        exit()

