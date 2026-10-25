#!/bin/bash -l
#SBATCH --job-name="dalia_shaheen3"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=<project_id>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=21
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

# Change to examples directory
if [[ "$(basename "$(pwd)")" != "examples" ]]; then
    echo ""
    echo "Error: Not in examples directory"
    echo "   Current directory: $(pwd)"
    echo "   Please run this script from the examples/ directory (under /scratch)"
    echo ""
    exit 1
fi

# --- Environment (see install.md, "On Shaheen III@KAUST")
module switch PrgEnv-cray PrgEnv-gnu
source $MY_SW/miniconda3-amd64/bin/activate $MY_SW/envs/dalia_hmpi_shaheen
source ../scripts/dalia_job_utils.sh && dalia_set_perfenv && dalia_print_job_config

# sTiles: the native library is downloaded on first import; compute nodes have
# no internet, so import it once on a login node with the same STILES_CACHE_DIR.
export STILES_CACHE_DIR=$MY_SW/stiles_cache
export STILES_NO_DOWNLOAD=1
export STILES_BIND=0
# One BLAS thread pool per rank of the size of the rank's CPU set.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- How to Run ---
# Shaheen III CPU nodes: 2x AMD Genoa, 192 physical cores, 384 GB. DALIA
# parallelises the 2 * n_hyperparameters + 1 function evaluations (9 for the
# gst models, 18 with the two Q factorizations of a Gaussian likelihood) over
# the MPI ranks; each rank runs its own sTiles factorization with
# --threads threads (default: 192 / ranks per node). Keep
# ntasks-per-node x cpus-per-task <= 192.

# --- GHCN-Daily temperature (prepare the inputs on a login node first) ---
srun --cpu-bind=cores python ./gst_ghcn/run.py --inputs full --solver stiles --threads ${SLURM_CPUS_PER_TASK} --max_iter 100

# --- Gaussian Spatio-temporal Model (Medium) Example, Serinv ---
# srun --cpu-bind=cores python ./gst_medium/run.py --solver_min_p 1 --max_iter 100

# --- Gaussian Regression Example ---
# srun --cpu-bind=cores python ./gr/run.py --max_iter 100
