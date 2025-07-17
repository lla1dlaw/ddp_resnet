import os
from typing import Union
from dotenv import load_dotenv
from s3torchconnector import S3MapDataset
from torchvision import Dataset

def _get_S3_stream():
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

def _load_saved_dataset():
    pass

def S1SLC_CVDL(*args, **kwargs) -> Union[S3MapDataset, Dataset]:
    """ Made for arbitrary usage as a replacement for PyTorch Datasets """
    download = kwargs.get('download', False) == True

    if download:
        return _get_S3_stream()
    else: 
        return _load_saved_dataset()

    print("Loaded Dataset.")
    return dataset


if __name__ == "__main__":
    # test download and see info about dataset.
    dataset = S1SLC_CVDL(root='./data', train=True, download=True, transform=None)
    sample = dataset[0].read()
    sample_type = type(sample)
    sample_dtype = sample.dtype()
    sample_dims = sample.shape()

    print(f"Sample Type: {sample_type}")
    print(f"Sample DType: {sample_dtype}")
    print(f"Sample Shape: {sample_dims}")
    print(f"\nSample: {sample}")
