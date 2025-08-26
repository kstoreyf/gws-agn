#!/bin/bash -l
set -euo pipefail
# Master script to run the full GW-AGN inference workflow
# This script submits jobs in the correct order with dependencies

echo "Starting GW-AGN inference workflow..."

# Create logs directory
mkdir -p logs

# Step 1: Submit preprocessing job
echo "Submitting preprocessing job..."
PREPROC_JOB=$(sbatch slurm_preprocessing.sh | awk '{print $4}')
echo "Preprocessing job ID: $PREPROC_JOB"

# Step 2: Submit GW sample generation job (depends on preprocessing)
echo "Submitting GW sample generation job..."
GW_SAMPLES_JOB=$(sbatch --dependency=afterok:$PREPROC_JOB slurm_gw_samples.sh | awk '{print $4}')
echo "GW samples job ID: $GW_SAMPLES_JOB"

# Step 3: Submit inference job (depends on both preprocessing and GW samples)
echo "Submitting inference job..."
INFERENCE_JOB=$(sbatch --dependency=afterok:$PREPROC_JOB:$GW_SAMPLES_JOB slurm_inference.sh | awk '{print $4}')
echo "Inference job ID: $INFERENCE_JOB"

echo ""
echo "Workflow submitted successfully!"
echo "Job IDs:"
echo "  Preprocessing: $PREPROC_JOB"
echo "  GW Samples: $GW_SAMPLES_JOB"
echo "  Inference: $INFERENCE_JOB"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in the logs/ directory" 