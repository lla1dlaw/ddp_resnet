#!/bin/bash
srun \
  --nodes=1 \
  --partition=gpu-l40 \
  --gpus-per-node=4 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --pty ./launch.sh
