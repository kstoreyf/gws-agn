#!/bin/bash
##SBATCH --job-name=inference_mcmc_nside256_nsteps500_Dz0.0001_betaH0_vary-H0
#SBATCH --job-name=inference_mcmc_seed1_nside64_dLunc1.0_nsteps500_Dz0.03_betaH0_vary-H0
#SBATCH --output=logs/%x.out
##SBATCH --time=0:30:00
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G #5G hits oom (was 2G, idk)
### perl
## SBATCH --constraint=gpu
## SBATCH --gres=gpu:1
## SBATCH --qos=regular
## SBATCH --account=cosmosim
#module load conda
### s3df
#SBATCH --partition=ampere
#SBATCH --qos=preemptable
#SBATCH --account=kipac:default
#SBATCH --gres=gpu:1

config_inf="../configs/configs_inference/config_inference_seed1_ratioNgalNagn1_bgal1.0_bagn1.0_nside64_seedgw1001_fagn0.0_lambdaagn0.0_zmaxgw1.0_dLunc1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0_vary-H0.yaml"
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn10_bgal1.0_bagn1.0_fagn0.5_lambdaagn0.0_mcmc_nw32_nsteps500_vary-alphaagn.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_fagn0.0_lambdaagn0.0_mcmc_nw32_nsteps500_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside64_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0try3_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside64_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0tryorig_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.03_betaH0_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_mcmc_nw32_nsteps500_Dz0.0001_betaH0_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_dLunc0.0_mcmc_nw32_nsteps500_Dz0.0001_betaH0_vary-H0.yaml'
#config_inf='../configs/configs_inference/config_inference_seed42_ratioNgalNagn1_bgal1.0_bagn1.0_nside256_fagn0.0_lambdaagn0.0_zmaxgw1.0_dLunc0.0_mcmc_nw32_nsteps500_Dz0.03_betaH0_vary-H0.yaml'


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
echo "Running on host: $(hostname)"
nvidia-smi || true

# Robust GPU preflight: don't rely on CUDA_VISIBLE_DEVICES being set.
python - <<'PY'
import sys
import jax

devices = jax.devices()
backend = jax.default_backend()
print("Preflight JAX devices:", devices)
print("Preflight JAX backend:", backend)

if backend != "gpu":
    print("ERROR: JAX is not on GPU before inference; aborting to avoid slow CPU fallback.")
    sys.exit(1)
PY

# Run the inference notebook
python run_inference.py --config $config_inf

echo "Inference job completed!" 