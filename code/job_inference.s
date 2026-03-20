#!/bin/bash
##SBATCH --job-name=inference_mcmc_nside256_nsteps500_Dz0.0001_betaH0_vary-H0
#SBATCH --job-name=inference_mcmc_nside256_dLunc0.0_nsteps500_Dz0.03_betaH0_vary-H0
#SBATCH --output=logs/%x.out
#SBATCH --time=0:30:00
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=5G #2G hits OOM
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=regular
#SBATCH --account=cosmosim

#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn10_bgal1.0_bagn1.0_fagn0.5_lambdaagn0.0_mcmc_nw32_nsteps500_vary-alphaagn.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_fagn0.0_lambdaagn0.0_mcmc_nw32_nsteps500_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside64_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0try3_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside64_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0tryorig_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.0001_betaH0_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_dLunc0.0_mcmc_nw32_nsteps500_Dz0.0001_betaH0_vary-H0.yaml'
config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_dLunc0.0_mcmc_nw32_nsteps500_Dz0.03_betaH0_vary-H0.yaml'

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

# Run inference
echo "Starting inference job..."
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

# Run the inference notebook
python run_inference.py --config $config_inf

echo "Inference job completed!" 