#!/bin/bash
#SBATCH --job-name=gws_preprocessing
#SBATCH --output=logs/preprocessing_%j.out
#SBATCH --error=logs/preprocessing_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=desi

# Load modules
module load conda
# Initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
conda activate gwsagn

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=""  # Disable GPU for preprocessing

# Run preprocessing
echo "Starting preprocessing job..."
python code/run_preprocessing.py

echo "Preprocessing job completed!" 