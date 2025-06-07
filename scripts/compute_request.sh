#!/bin/bash

# Exit on error
set -e

# set python path
export PYTHONPATH="$PYTHONPATH:$PWD/mma2025"

# request compute
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=1 --time=00:20:00 --pty bash -i

# activate venv
source .venv/bin/activate

# run main app
python src/main.py
