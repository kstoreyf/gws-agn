#!/bin/bash
#SBATCH --job-name=test
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cpu

echo "Test job running"
sleep 10
echo "Test job completed" 