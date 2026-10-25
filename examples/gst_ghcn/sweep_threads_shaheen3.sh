#!/bin/bash
# Full-inference wall time of the GHCN example with the sTiles solver against
# its thread count, one MPI process, one Shaheen III node per job (exclusive,
# so the measurements do not share the node). Usage, from this directory on a
# login node, after `prepare_inputs.py --name full`:
#
#     ./sweep_threads_shaheen3.sh <project_id> [inputs_name] [thread counts...]
#
# Default thread counts: 1 2 4 8 16 32 48 96 192 (8 = one CCD sharing an L3,
# 96 = one socket, 192 = the node). Each job writes
# outputs_<inputs>_stiles_r1_t<threads>_<jobid>/summary.json; collect them with
# `python collect_sweep.py` into stiles_fullsolve.csv for the poster figure.
set -e
ACCOUNT=${1:?project id}
INPUTS=${2:-full}
shift 2 2>/dev/null || shift $#
THREADS_LIST=${@:-1 2 4 8 16 32 48 96 192}
HERE=$(cd "$(dirname "$0")" && pwd)

for T in $THREADS_LIST; do
    # generous limits: the 1-thread run of the full case can take several hours
    if   [ "$T" -le 2 ]; then WALL=08:00:00
    elif [ "$T" -le 8 ]; then WALL=04:00:00
    else WALL=02:00:00; fi
    sbatch <<SLURM
#!/bin/bash -l
#SBATCH --job-name="ghcn_t${T}"
#SBATCH --output=${HERE}/%x.%j.out
#SBATCH --error=${HERE}/%x.%j.err
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${T}
#SBATCH --hint=nomultithread
#SBATCH --time=${WALL}

module switch PrgEnv-cray PrgEnv-gnu
source \$MY_SW/miniconda3-amd64/bin/activate \$MY_SW/envs/dalia_hmpi_shaheen
export STILES_CACHE_DIR=\$MY_SW/stiles_cache STILES_NO_DOWNLOAD=1 STILES_BIND=0
export OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T}
cd ${HERE}
echo "config: 1 rank x ${T} threads, inputs ${INPUTS}, node \$(hostname)"
srun --cpu-bind=cores python run.py --inputs ${INPUTS} --solver stiles --threads ${T} --max_iter 100 --no-reference --tag \${SLURM_JOB_ID}
SLURM
done
