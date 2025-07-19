#!/bin/bash

echo "🚀 Requesting interactive GPU session from SLURM..."

# Define the sequence of commands to be run on the allocated nodes.
COMMANDS=$(
  cat <<'EOF'
# This block runs inside the SLURM allocation for each task

# 1. Load necessary modules
module purge
module load cudnn8.5-cuda11.7/8.5.0.96
module load conda
module load slurm
echo "Modules loaded by process with SLURM_PROCID=$SLURM_PROCID"

# 2. Activate Conda environment
conda activate FederatedResnet

# 3. Set up environment variables for DDP
# Translate SLURM variables to what PyTorch expects.
# srun ensures each process gets a unique ID.
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=$SLURM_LOCALID

# This is the key for preventing GPU conflicts
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Process $RANK of $WORLD_SIZE (Local Rank: $LOCAL_RANK) starting on GPU $CUDA_VISIBLE_DEVICES"

# 4. Launch the training script directly
python train.py --epochs 200 --batch_size 64 --dataset cifar10 --model-type real

EOF
)

# Execute the command sequence within an interactive srun session
# We now ask srun for 4 tasks, and it will run the COMMANDS string for each one.
srun \
  --nodes=1 \
  --partition=gpu-l40 \
  --gpus-per-node=4 \
  --ntasks-per-node=4 \
  --cpus-per-task=12 \
  --pty \
  bash -c "$COMMANDS"

echo "--- ✅ Interactive session has ended. ---"
