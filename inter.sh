#!/bin/bash

srun --nodes=1 --partition=gpu-l40 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=12 --pty bash -i
