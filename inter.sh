#!/bin/bash

COMMANDS=$(
  cat <<'EOF'
module purge
module load cudnn8.5-cuda11.7/8.5.0.96
module load conda 
mdule load slurm
echo "Modules loaded."

conda init bash
conda activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# --- Set up environment for torchrun ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355 # A static port is fine for single-node jobs

torchrun \
  --standalone \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  train.py --epochs 200 --batch_size 64 --dataset cifar10 --model-type real

EOF
)
srun \
  --nodes=1 \
  --partition=gpu-l40 \
  --gpus-per-node=4 \
  --ntasks-per-node=1 \
  --cpus-per-task=12 \
  --pty \
  bash -c "$COMMANDS"
