#!/bin/bash -l

# sbatch documentation: https://slurm.schedmd.com/sbatch.html

# SLURM SUBMIT SCRIPT
# Rules:
#   Set --nodes to the number of nodes
#   Set --gres=gpu:X,VRAM:24G and --ntasks-per-node=X to the same number
#   Set --mem to about 10*X (e.g. 20G for two GPUs per node, if your job needs 10GB of RAM per GPU)
#   This will give nodes*X total GPUs

#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --output=./logs/output.log
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

# --- Setup In Master ---
# Master will be the one running this bash script (SLURM runs this only once)
# Get hostname and port on first node first process
# For the port see: https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
master_addr=$(hostname -i)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

module purge
module load mamba
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

# --- Call the Script which the User will Edit ---
# With srun this will be run on all nodes for all processes
srun --gres=gpu:1 ./distributed_data_parallel_slurm_run.bash --master_addr $master_addr --master_port $master_port
