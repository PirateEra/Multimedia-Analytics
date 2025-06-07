#!/bin/bash

# Exit on error
set -e

cd mma2025

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Virtual environment is active."
