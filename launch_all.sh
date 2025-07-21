#!/bin/bash

# This script sets up the environment and executes torchrun commands in parallel on 4 GPUs.

echo "--- Optimized 4-GPU Launch Script Started on host: $(hostname) ---"

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

# 3. Launch training scripts in parallel, two at a time, each on 2 GPUs.
echo "--- Launching HH scripts in parallel (2x 2-GPU jobs) ---"
# Run 'real' model on GPUs 0,1 and 'complex' on GPUs 2,3, assigning unique ports

# Job 1 on Port 29500
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29500 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL_HH --trials 5 --model-type real &

# Job 2 on Port 29501
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29501 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL_HH --trials 5 --model-type complex &

# Wait for the two HH training jobs to finish before proceeding
wait
echo "--- HH runs complete ---"

echo "--- Launching HV scripts in parallel (2x 2-GPU jobs) ---"
# Job 3 on Port 29500
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29500 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL_HV --trials 5 --model-type real &

# Job 4 on Port 29501
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29501 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL_HV --trials 5 --model-type complex &

wait
echo "--- HV runs complete ---"

echo "--- Launching base CVDL scripts in parallel (2x 2-GPU jobs) ---"
# Job 5 on Port 29500
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29500 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL --trials 5 --model-type real &

# Job 6 on Port 29501
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=8 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --rdzv-endpoint=localhost:29501 \
  ./train.py --epochs 100 --batch_size 32 --dataset S1SLC_CVDL --trials 5 --model-type complex &

wait
echo "--- All training runs complete ---"
