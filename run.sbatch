#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu-l40
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1  # Very important: We only start ONE task, which is the launch.sh script
#SBATCH --cpus-per-task=12
#SBATCH --output=./logs/sbatch_output_%j.log # %j adds the job ID to the log name
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

# Execute the main launch script
./launch.sh
