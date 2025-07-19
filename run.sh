#!/bin/bash -l

# sbatch documentation: https://slurm.schedmd.com/sbatch.html

# SLURM SUBMIT SCRIPT
# Rules:
#    Set --nodes to the number of nodes
#    Set --gpus-per-node to the number of GPUs you want to use on each node
#    Set --ntasks-per-node to be equal to --gpus-per-node
#    Set --mem to about 10*X (e.g. 20G for two GPUs per node, if your job needs 10GB of RAM per GPU)
#    This will give nodes*X total GPUs

#SBATCH --nodes=1
#SBATCH --partition=gpu-l40
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --output=./logs/output.log
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

module purge
module load mamba
module load slurm
module load cudnn8.5-cuda11.7/8.5.0.96
module load gcc/10.2.0
echo "Modules loaded."
# 2. Activate your Conda environment
mamba activate FederatedResnet
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
echo "Python path: $(which python)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo "---------------------------"

# --- Set up environment for DDP ---
# Get the master node's IP address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Get a free port
export MASTER_PORT=12355

# --- Call the Script which the User will Edit ---
# Use srun to launch the parallel tasks.
# SLURM will automatically set environment variables like SLURM_PROCID (rank).
srun python train.py --epochs 5 --batch_size 128
