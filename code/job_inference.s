#!/bin/bash
##SBATCH --job-name=inference_mcmc
#SBATCH --job-name=inference_like
#SBATCH --output=logs/%x.out
##SBATCH --time=0:20:00
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=regular
#SBATCH --account=desi

# Load modules
module load conda
# Initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
conda activate glassenv

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# # Check if preprocessing files exist
# if [ ! -f "lognormal_pixelated_nside_64_galaxies.h5" ] || [ ! -f "lognormal_pixelated_nside_64_agn.h5" ]; then
#     echo "Error: Preprocessing files not found!"
#     echo "Please run preprocessing first: sbatch slurm_preprocessing.sh"
#     exit 1
# fi

# Run inference
echo "Starting inference job..."
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

# Run the inference notebook
python run_inference.py

echo "Inference job completed!" 