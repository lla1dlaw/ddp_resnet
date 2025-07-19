#!/bin/bash

module purge
module load slurm
module load cudnn8.5-cuda11.7/8.5.0.96
module load gcc/10.2.0
echo "Modules loaded."

mamba activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"
srun --nodes=1 --partition=gpu-l40 --gpus-per-node=4 --ntasks-per-node=1 --cpus-per-task=12 --pty bash -i
