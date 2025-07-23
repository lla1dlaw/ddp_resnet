#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu-l40
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=./logs/sbatch_output_%j.log # %j adds the job ID to the log name
#SBATCH --mail-user=liamlaidlaw04@gmail.com
#SBATCH --mail-type=ALL

# Execute the main launch script
./launch_HH.sh
