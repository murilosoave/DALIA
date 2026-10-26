#!/bin/bash
# One MPI process per Shaheen III node, 96 sTiles threads each (one Genoa
# socket), for the GHCN example: 1x96, 3x96, 9x96 as ranks x threads, i.e.
# 1, 3 and 9 nodes. Same configurations as a within-node run but with every
# rank on its own node, so the ranks do not share memory bandwidth, L3 or the
# 384 GB of one node; what changes between the two is the cost of the
# inter-node MPI traffic of each L-BFGS iteration. Usage, from this directory
# on a login node, after `prepare_inputs.py --name full`:
#
#     ./sweep_nodes_shaheen3.sh <project_id> [inputs_name] [solver] [node counts...]
#
# Default node counts: 1 3 9 (9 = the 2 * n_hyperparameters + 1 function
# evaluations of an iteration, one per rank). The optional third argument runs
# the same configurations with another solver, e.g. `serinv` for the baseline.
# Each job writes
# outputs_<inputs>_<solver>_r<nodes>_t96_nodes<nodes>_<jobid>/summary.json
# (the `nodes` tag tells these runs apart from the single-node ones of
# sweep_ranks_shaheen3.sh and sweep_threads_shaheen3.sh, which share the
# ranks x threads name); collect them with `python collect_sweep.py`.
set -e
ACCOUNT=${1:?project id}
INPUTS=${2:-full}
SOLVER=${3:-stiles}
shift 3 2>/dev/null || shift $#
NODES_LIST=${@:-1 3 9}
THREADS=96
HERE=$(cd "$(dirname "$0")" && pwd)

for NODES in $NODES_LIST; do
    if [ "$SOLVER" = "stiles" ]; then
        SOLVER_ARGS="--solver stiles --threads ${THREADS}"
    else
        SOLVER_ARGS="--solver ${SOLVER}"
    fi
    # the single-process run does all evaluations in sequence: longer limit
    if [ "$NODES" -le 1 ]; then WALL=06:00:00; else WALL=03:00:00; fi
    sbatch <<SLURM
#!/bin/bash -l
#SBATCH --job-name="ghcn_${SOLVER}_${NODES}nodes_x${THREADS}"
#SBATCH --output=${HERE}/%x.%j.out
#SBATCH --error=${HERE}/%x.%j.err
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${THREADS}
#SBATCH --hint=nomultithread
#SBATCH --time=${WALL}

module switch PrgEnv-cray PrgEnv-gnu
source \$MY_SW/miniconda3-amd64/bin/activate \$MY_SW/envs/dalia_hmpi_shaheen
export STILES_CACHE_DIR=\$MY_SW/stiles_cache STILES_NO_DOWNLOAD=1 STILES_BIND=0
export OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS}
cd ${HERE}
echo "config: ${NODES} nodes x 1 rank x ${THREADS} threads, solver ${SOLVER}, inputs ${INPUTS}, nodes \$SLURM_JOB_NODELIST"
srun --cpu-bind=cores python run.py --inputs ${INPUTS} ${SOLVER_ARGS} --max_iter 100 --no-reference --tag nodes${NODES}_\${SLURM_JOB_ID}
SLURM
done
