#!/bin/bash

# This script sets up the environment and executes the torchrun command.
# It can be called by both sbatch and srun.

echo "--- Launch Script Started on host: $(hostname) ---"

# 1. Load necessary modules
module purge
module load cudnn8.5-cuda11.7/8.5.0.96
module load conda
module load slurm
module load gcc/10.2.0
echo "Modules loaded."

# 2. Activate Conda environment
conda activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Launch the training script with torchrun
# torchrun will spawn a process for each GPU and set LOCAL_RANK correctly.
echo "--- Launching training script via torchrun ---"

OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --log_dir=./torch_logs \
  ./train.py --epochs 2 --batch_size 64 --dataset S1SLC_CVDL --trials 1 --model-type real
