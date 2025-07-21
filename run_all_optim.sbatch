#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/sbatch_output_%j.log # %j adds the job ID to the log name
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

# Execute the main optimized launch script for 4 GPUs
./launch_all.sh
