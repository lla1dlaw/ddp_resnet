#!/bin/bash
srun \
  --nodes=1 \
  --partition=gpu-l40 \
  --gres=gpu:4 \
  --ntasks-per-node=1 \
  --cpus-per-task=12 \
  --pty ./launch.sh
