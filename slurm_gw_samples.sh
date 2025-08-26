#!/bin/bash
#SBATCH --job-name=gws_gw_samples
#SBATCH --output=logs/gw_samples_%j.out
#SBATCH --error=logs/gw_samples_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
export CUDA_VISIBLE_DEVICES=""  # Disable GPU for GW sample generation

# Check if preprocessing files exist
if [ ! -f "lognormal_pixelated_nside_64_galaxies.h5" ] || [ ! -f "lognormal_pixelated_nside_64_agn.h5" ]; then
    echo "Error: Preprocessing files not found!"
    echo "Please run preprocessing first: sbatch slurm_preprocessing.sh"
    exit 1
fi

# Run GW sample generation
echo "Starting GW sample generation..."
jupyter nbconvert --to notebook --execute notebooks/inference/generate_gwsamples.ipynb --output gw_samples_results.ipynb

echo "GW sample generation completed!" 