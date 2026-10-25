#!/bin/bash
# Ranks x threads on one Shaheen III node for the GHCN example: how to fill a
# node with a shared-memory solver. Usage, from this directory on a login node,
# after `prepare_inputs.py --name full`:
#
#     ./sweep_ranks_shaheen3.sh <project_id> [inputs_name] [solver]
#
# Configurations (ranks x threads = 192 physical cores): 1x192, 2x96, 6x32,
# 9x21, 12x16. DALIA runs the 2 * n_hyperparameters + 1 = 9 function
# evaluations of an iteration on separate ranks (18 with the two Q
# factorizations of the Gaussian likelihood), each rank factorizing with its
# own sTiles threads. Memory bounds the rank count: one process of the full
# case peaks at ~25 GB (sTiles symbolic analysis, simultaneous on all ranks),
# so 12 ranks (~300 GB) is the most that fits in the 384 GB of a node; 18x10
# would be killed at the first evaluation.
#
# The optional third argument runs the same configurations with another
# solver, e.g. `serinv` for the baseline (same memory order of magnitude).
# Each job writes outputs_<inputs>_<solver>_r<ranks>_t<threads>_<jobid>/summary.json;
# collect them with `python collect_sweep.py [--solver serinv]`.
set -e
ACCOUNT=${1:?project id}
INPUTS=${2:-full}
SOLVER=${3:-stiles}
HERE=$(cd "$(dirname "$0")" && pwd)

for cfg in "1 192" "2 96" "6 32" "9 21" "12 16"; do
    set -- $cfg
    RANKS=$1; THREADS=$2
    if [ "$SOLVER" = "stiles" ]; then
        SOLVER_ARGS="--solver stiles --threads ${THREADS}"
    else
        SOLVER_ARGS="--solver ${SOLVER}"
    fi
    # the single-process run does all evaluations in sequence: longer limit
    if [ "$RANKS" -le 2 ]; then WALL=04:00:00; else WALL=02:00:00; fi
    sbatch <<SLURM
#!/bin/bash -l
#SBATCH --job-name="ghcn_${SOLVER}_${RANKS}x${THREADS}"
#SBATCH --output=${HERE}/%x.%j.out
#SBATCH --error=${HERE}/%x.%j.err
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${RANKS}
#SBATCH --cpus-per-task=${THREADS}
#SBATCH --hint=nomultithread
#SBATCH --time=${WALL}

module switch PrgEnv-cray PrgEnv-gnu
source \$MY_SW/miniconda3-amd64/bin/activate \$MY_SW/envs/dalia_hmpi_shaheen
export STILES_CACHE_DIR=\$MY_SW/stiles_cache STILES_NO_DOWNLOAD=1 STILES_BIND=0
export OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS}
cd ${HERE}
echo "config: ${RANKS} ranks x ${THREADS} threads, solver ${SOLVER}, inputs ${INPUTS}, node \$(hostname)"
srun --cpu-bind=cores python run.py --inputs ${INPUTS} ${SOLVER_ARGS} --max_iter 100 --no-reference --tag \${SLURM_JOB_ID}
SLURM
done
