#!/bin/bash -l
set -euo pipefail
# Local preprocessing script for testing

echo "Running preprocessing locally..."

# Load modules
module load conda
# Initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
conda activate gwsagn

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=""  # Disable GPU for preprocessing

# Run preprocessing
echo "Starting preprocessing job..."
python code/run_preprocessing.py

echo "Preprocessing job completed!" 