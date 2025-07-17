import pretty_errors
import os
import numpy as np
from pathlib import Path
from sys import base_exec_prefix
from typing import Union, Optional, Callable, Any
from dotenv import load_dotenv
from s3torchconnector import S3MapDataset
from torch.utils.data.dataset import Dataset
from ProgressFile import ProgressFile


def S1SLC_CVDL( # Call this method only.
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
) -> Union[S3MapDataset, Dataset]: 

    """ Made for arbitrary usage as a replacement for PyTorch Datasets """

    if download:
        return _get_S3_stream()
    else: 
        base_dir = "S1SLC_CVDL"
        return _load_saved_dataset(root_dir=root, base_dir=base_dir)


def _get_S3_stream() -> S3MapDataset:
    URI = "s3://ieee-dataport/open/98396/S1SLC_CVDL.rar"
    REGION = "us-east-1"
    try:
        load_dotenv() # AWS secret ID & Secret key must be stored in a .env file for accessing the SLSLC_CVDL. If such a file does not exist, create one.
        return S3MapDataset.from_prefix(URI, region=REGION) # S3MapDataset streams data from an S3 bucket rather than downloading the full dataset. 
    except FileNotFoundError:
        print(".env file not found in project root directory and is required for download.")
        print("Create this file in the project root and place your IEEE-dataport AWS secret ID and secret key inside with the following format:")
        print("\tAWS_ACCESS_KEY_ID=<your ID>\n\tAWS_SECRET_ACCESS_KEY=<your secret key>")
        print("Or remove 'download = True' argument from S1SLC_CVDL call to attempt local loading.")
        exit()


def _load_saved_dataset(root_dir: str, base_dir:str) -> Dataset:
    path = os.path.join(root_dir, base_dir)
    try:
        os.path.exists(path)
    except FileNotFoundError:
        print(f"Could not find directory: {path}")
    HH_data = []
    HV_data = []
    label_data = []

    for dir in os.listdir(path):
        data_dir = os.path.join(path, dir)

        HH_path = os.path.join(data_dir, 'HH_Complex_Patches.npy')
        HV_path = os.path.join(data_dir, 'HV_Complex_Patches.npy')
        labels_path = os.path.join(data_dir, 'Labels.npy')

        print(f"\nLoading S1SLC_CVDL {dir}")
        with ProgressFile(HH_path, "rb", desc=f'reading {HH_path}') as f:
            HH = np.load(f)
            f.close()
        with ProgressFile(HH_path, "rb", desc=f'reading {HV_path}') as f:
            HV = np.load(f)
            f.close()
        with ProgressFile(HH_path, "rb", desc=f'reading {labels_path}') as f:
            labels = np.load(f)
            f.close()
        
        print("\n" + "="*30)
        print(f"\n{dir} HH Shape: {HH.shape}")
        print(f"{dir} HH dtype: {HH.dtype}")
        print(f"\n{dir} HV Shape: {HV.shape}")
        print(f"{dir} HV dtype: {HV.dtype}")
        print(f"\n{dir} Labels Shape: {labels.shape}")
        print(f"{dir} Labels dtype: {labels.dtype}")

        HH_data.append(HH)
        HV_data.append(HV)
        label_data.append(labels)

    HH_data = np.array(HH_data)
    HV_data = np.array(HV_data)
    label_data = np.array(label_data)


if __name__ == "__main__":
    _load_saved_dataset("./data", "S1SLC_CVDL")
