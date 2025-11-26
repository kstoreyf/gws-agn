#!/bin/bash
set -euo pipefail

# Slurm-backed ipykernel launcher for NERSC (Perlmutter)
#
# This script requests an interactive allocation and starts an ipykernel
# on a compute node with your chosen resources. You can then attach your
# existing Jupyter notebook to the kernel via the connection file printed.
#
# Configuration via environment variables (override as needed):
#   TIME=04:00:00           # walltime
#   CONSTRAINT=cpu          # node type (cpu/gpu)
#   QOS=interactive         # QoS for interactive jobs
#   NODES=1                 # number of nodes
#   CPUS_PER_TASK=32        # CPU cores for your kernel process
#   GPUS_PER_TASK=0         # number of GPUs per task (0 = no GPU)
#   MEM=20G                   # memory per node (0 = all memory; omitted on GPU)
#   MODE=kernel              # kernel | server
#   ACCOUNT=""               # optional Slurm account/project to charge (e.g., m1234)
#   ENV_ACTIVATE=""         # path to source your env (e.g. "$HOME/mambaforge/bin/activate myenv")
#   CONN_DIR="$HOME/.local/share/jupyter/runtime/remote-kernels"
#   LOG_DIR="$PWD/logs"       # where to write srun logs
#   DEBUG=0                  # 1 to enable verbose/debug env
#
# Usage examples:
#   ./slurm_start_ipykernel.sh
#   TIME=08:00:00 CPUS_PER_TASK=64 MEM=0 ./slurm_start_ipykernel.sh
#   ENV_ACTIVATE="$HOME/mambaforge/bin/activate myenv" ./slurm_start_ipykernel.sh

TIME=${TIME:-04:00:00}
CONSTRAINT=${CONSTRAINT:-cpu}
QOS=${QOS:-interactive}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
GPUS_PER_TASK=${GPUS_PER_TASK:-0}
MEM=${MEM:-0}
CONN_DIR=${CONN_DIR:-"$HOME/.local/share/jupyter/runtime/remote-kernels"}
ENV_ACTIVATE=${ENV_ACTIVATE:-""}
MODE=${MODE:-kernel}
ACCOUNT=${ACCOUNT:-""}
LOG_DIR=${LOG_DIR:-"$PWD/logs"}
DEBUG=${DEBUG:-0}

# GPU-aware flags
if [[ "$GPUS_PER_TASK" != "0" ]]; then
  # If user left default cpu constraint but asked for GPUs, prefer gpu
  if [[ "$CONSTRAINT" == "cpu" ]]; then
    CONSTRAINT=gpu
  fi
  EXTRA_ALLOC_FLAGS="-G $GPUS_PER_TASK"
  EXTRA_SRUN_FLAGS="--gpus-per-task=$GPUS_PER_TASK --gpu-bind=closest"
  # On GPU interactive jobs, omit explicit --mem to satisfy policy
  MEM_FLAG=""
else
  EXTRA_ALLOC_FLAGS=""
  EXTRA_SRUN_FLAGS=""
  # Only set MEM_FLAG if MEM is non-empty and not "0"
  if [[ -n "$MEM" && "$MEM" != "0" ]]; then
    MEM_FLAG="--mem=$MEM"
  else
    MEM_FLAG=""
  fi
fi

# Optional account flag
if [[ -n "$ACCOUNT" ]]; then
  ACCOUNT_FLAG="-A $ACCOUNT"
else
  ACCOUNT_FLAG=""
fi

mkdir -p "$CONN_DIR"
mkdir -p "$LOG_DIR"

if [[ "$DEBUG" == "1" ]]; then
  set -x
fi

echo "Requesting interactive allocation: NODES=$NODES, CONSTRAINT=$CONSTRAINT, QOS=$QOS, TIME=$TIME, MEM=$MEM, GPUS_PER_TASK=$GPUS_PER_TASK, MODE=$MODE, ACCOUNT=${ACCOUNT:-none}"

# We use salloc for the allocation and then run a single srun that starts the ipykernel.
# The kernel will keep the allocation busy until you shut the kernel down from Jupyter
# or interrupt the job.
salloc -N "$NODES" -C "$CONSTRAINT" -q "$QOS" -t "$TIME" $ACCOUNT_FLAG $EXTRA_ALLOC_FLAGS ${MEM_FLAG} bash -lc "
  set -euo pipefail
  echo 'Allocated host: ' \"\$(hostname)\"
  echo 'SLURM_JOB_ID: ' \"\${SLURM_JOB_ID:-unknown}\"

  CONN_DIR=\"$CONN_DIR\"
  mkdir -p \"\$CONN_DIR\"
  CONN_FILE=\"\$CONN_DIR/kernel-\${SLURM_JOB_ID:-$RANDOM}.json\"

  if [[ -n \"$ENV_ACTIVATE\" ]]; then
    # shellcheck disable=SC1090
    source \"$ENV_ACTIVATE\"
  fi

  # Helpful env for stability and debugging on HPC
  export PYTHONFAULTHANDLER=1
  export HDF5_USE_FILE_LOCKING=FALSE
  export MALLOC_TRIM_THRESHOLD_=0
  export OMP_NUM_THREADS=\"$CPUS_PER_TASK\"
  export MKL_THREADING_LAYER=GNU
  export MKL_NUM_THREADS=\"$CPUS_PER_TASK\"
  export OPENBLAS_NUM_THREADS=\"$CPUS_PER_TASK\"
  export NUMEXPR_MAX_THREADS=\"$CPUS_PER_TASK\"
  ulimit -n 65536 || true

  OUT=\"$LOG_DIR/interactive_\${SLURM_JOB_ID}_out.log\"
  ERR=\"$LOG_DIR/interactive_\${SLURM_JOB_ID}_err.log\"
  echo \"Logs: \$OUT\nErrs: \$ERR\"

  case \"$MODE\" in
    kernel)
      if ! python -c 'import ipykernel' >/dev/null 2>&1; then
        echo 'ERROR: ipykernel is not installed in this environment.'
        echo 'Install it, e.g.: pip install ipykernel  (or)  mamba install ipykernel'
        exit 1
      fi

      echo
      echo 'Starting ipykernel on compute node via srun...'
      echo 'Connection file:' \"\$CONN_FILE\"
      echo
      echo 'Attach from your editor: Select Existing Kernel and choose the file above.'
      echo

      srun -n 1 -c \"$CPUS_PER_TASK\" --cpu-bind=cores $EXTRA_SRUN_FLAGS -o \"\$OUT\" -e \"\$ERR\" python -m ipykernel_launcher -f \"\$CONN_FILE\"
      ;;
    server)
      if ! command -v jupyter >/dev/null 2>&1; then
        echo 'ERROR: jupyter is not installed in this environment.'
        echo 'Install it, e.g.: pip install jupyterlab  (or)  mamba install jupyterlab'
        exit 1
      fi

      PORT=\"\${PORT:-8888}\"
      echo
      echo 'Starting Jupyter Lab on compute node via srun (URL will appear below and in logs)...'
      HOST_FQDN=\"\$(hostname -f)\"
      HOST_IP=\"\$(hostname -I | awk '{print \$1}')\"
      echo '================================================================================'
      echo 'CONNECTION INSTRUCTIONS:'
      echo '================================================================================'
      echo 'If Cursor is on YOUR LOCAL MACHINE:'
      echo '  1. On your LOCAL machine, run:'
      echo \"     ssh -N -L \$PORT:\${HOST_FQDN}:\$PORT <nersc_username>@perlmutter-p1.nersc.gov\"
      echo '  2. Use URL: http://127.0.0.1:'\"\$PORT\"'/?token=...'
      echo
      echo 'If Cursor is on a NERSC LOGIN NODE (you are here):'
      echo '  Option A - Direct connection (usually works):'
      echo '    Use the URL as printed by Jupyter below (with compute node hostname)'
      echo '    Example: http://'\"\${HOST_FQDN}\":\"\$PORT\"'/?token=...'
      echo
      echo '  Option B - If Option A fails, use SSH tunnel from login node:'
      echo \"    ssh -N -L \$PORT:localhost:\$PORT \$(hostname)\"
      echo '    Then use: http://127.0.0.1:'\"\$PORT\"'/?token=...'
      echo '================================================================================'
      echo
      # Stream to terminal and log so you can copy the URL
      srun -n 1 -c \"$CPUS_PER_TASK\" --cpu-bind=cores $EXTRA_SRUN_FLAGS jupyter lab --no-browser --ip=0.0.0.0 --port=\"\$PORT\" --NotebookApp.allow_origin=* --ServerApp.allow_remote_access=True 2>&1 | tee -a \"\$OUT\"
      ;;
    *)
      echo 'Unknown MODE. Use MODE=kernel or MODE=server'
      exit 2
      ;;
  esac
"

echo
echo "Kernel exited and allocation ended."


