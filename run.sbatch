#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=gpu-l40
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --output=./logs/output.log
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

module purge
module load slurm
module load cudnn8.5-cuda11.7/8.5.0.96
module load gcc/10.2.0
echo "Modules loaded."

mamba activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# --- Set up environment for torchrun ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355 # A static port is fine for single-node jobs

torchrun \
  --standalone \
  --nproc_per_node=$SLURM_TASKS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  train.py --epochs 2 --batch_size 64
