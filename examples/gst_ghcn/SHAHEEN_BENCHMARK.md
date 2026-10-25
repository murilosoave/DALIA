# Running the sTiles benchmark on Shaheen III, step by step

Goal: the wall time of a full DALIA inference on the GHCN-Daily 2022 case
(456k latent variables, 4.31 M observations) with the sTiles solver, against
the number of threads (one process) and against the number of processes on one
node, plus the Serinv baseline. The result feeds `figs_dalia/stiles_threads.pdf`
of the KAUST poster.

Everything below runs on Shaheen III (HPE Cray EX, CPU nodes with 2x AMD
Genoa, 192 physical cores and 384 GB each). Conventions that shape every step:

- `/home` is not mounted on compute nodes and `/project` is read-only there,
  so code, data, conda environments and outputs live under `/scratch/$USER`.
  Software goes in `$MY_SW` (`/scratch/$USER/iops/sw`), which is not purged.
- Compute nodes have no internet: clone, download and install on login nodes.
- Every job is charged to a project: `#SBATCH --account=<project_id>`.
  `sacctmgr show assoc user=$USER format=account` lists the valid ids.

Shell set-up used in every step (put it in `~/.bashrc` or source it; the
environment it refers to is created in step 0):

```bash
module switch PrgEnv-cray PrgEnv-gnu
source $MY_SW/miniconda3-amd64/bin/activate $MY_SW/envs/dalia_hmpi_shaheen
export PATH=$HOME/.local/bin:$PATH                 # git-lfs
export STILES_CACHE_DIR=$MY_SW/stiles_cache        # must be under /scratch, not ~/.cache
export STILES_BIND=0
export DALIA=/scratch/$USER/src/DALIA
```

## 0. Install DALIA on Shaheen (once, login node)

Condensed from `install.md`, "On Shaheen III@KAUST". Skip what is already
in place.

Login: `ssh <username>@shaheen.hpc.kaust.edu.sa` (password + one-time code;
with `ControlMaster` in `~/.ssh/config` later sessions reuse the connection).

1. git-lfs in your user space (the example data are LFS-tracked):
    ```bash
    mkdir -p ~/.local/bin && cd $MY_SW
    curl -LO https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz
    tar xzf git-lfs-linux-amd64-v3.6.1.tar.gz && cp git-lfs-3.6.1/git-lfs ~/.local/bin/
    export PATH=$HOME/.local/bin:$PATH
    ```
2. Clone the repositories:
    ```bash
    mkdir -p /scratch/$USER/src && cd /scratch/$USER/src
    git clone https://github.com/murilosoave/DALIA && (cd DALIA && git lfs install && git lfs pull)
    git clone https://github.com/vincent-maillou/serinv
    ```
3. Miniconda in the high-IOPS tier (the installer is on the system):
    ```bash
    mkdir -p $MY_SW && cd $MY_SW
    bash /sw/sources/miniconda/conda24.1.2-python3.12.1/Miniconda3-latest-Linux-x86_64.sh -b -s -p $MY_SW/miniconda3-amd64 -u
    source $MY_SW/miniconda3-amd64/bin/activate
    conda install -y -c conda-forge mamba==1.5.8
    ```
4. The base environment (x86 file; DALIA needs Python 3.13):
    ```bash
    module switch PrgEnv-cray PrgEnv-gnu
    mamba env create -f /scratch/$USER/src/DALIA/envs/dalia_base_x86.yml -p $MY_SW/envs/dalia_base_shaheen
    conda clean --all -y
    conda activate $MY_SW/envs/dalia_base_shaheen
    cd /scratch/$USER/src/DALIA && python -m pip install --no-deps --editable .
    cd /scratch/$USER/src/serinv && python -m pip install --no-deps --editable .
    ```
5. The MPI environment: clone it and build `mpi4py` against Cray MPICH
   through the compiler wrapper (never `conda install mpi4py`):
    ```bash
    conda deactivate
    conda create -p $MY_SW/envs/dalia_hmpi_shaheen --clone $MY_SW/envs/dalia_base_shaheen -y
    conda activate $MY_SW/envs/dalia_hmpi_shaheen
    MPICC=cc python -m pip install --no-cache-dir --no-binary mpi4py mpi4py
    conda clean --all -y
    python -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # must say CRAY MPICH
    ```
6. Test suite, from the `tests/` directory (it uses relative paths), inside an
   allocation on the debug partition:
    ```bash
    salloc --account=<project_id> --partition=debug --nodes=1 --time=00:30:00
    cd /scratch/$USER/src/DALIA/tests && ./runner.sh
    exit
    ```
   Expected: all CPU and MPI tests pass; the GPU parametrisations are skipped.

## 1. Get this branch onto Shaheen

The example and the benchmark scripts are on branch `bench/shareen_cluster`
of `github.com/murilosoave/DALIA` (commit `2d501b4` or later). On a login node:

```bash
cd $DALIA && git fetch && git checkout bench/shareen_cluster && git lfs pull
python -m pip install --no-deps --editable .     # refresh the editable install if the branch changed
ls examples/gst_ghcn/inputs_small/inputs_spatio_temporal/   # c0 g1 g2 g3 m0 m1 m2 a .npz
```

## 2. Install sTiles in the environment (login node, internet needed)

The `stiles` wheel is pure Python; the native `libstiles.so` is downloaded
from the GitHub releases of `github.com/esmail-abdulfattah/sTiles` on the
first import and cached. Use the most recent release (the wheel and the
native library share the version; `Version_2026.10.19` at the time of
writing) and record the version in the benchmark notes:

```bash
python -m pip install -U stiles
mkdir -p $STILES_CACHE_DIR
python -c "import sTiles; print(sTiles.__version__)"
ls $STILES_CACHE_DIR                                  # one Version_<date> directory, same date as the wheel
# if the directory is older than the wheel (a stale cache), force the download:
python -m sTiles update
find $STILES_CACHE_DIR -name 'libstiles.so'           # under .../libstiles-linux-x86_64/
```

The import resolves the newest release tag through the GitHub API and
downloads the matching asset, so both calls need internet (login node). To
pin a specific release later, for reproducibility, export
`STILES_RELEASE_TAG=Version_<date>` before the first import and install the
wheel of the same date.

Notes:

- The x86_64 asset is the AVX2 build with MKL embedded. Genoa supports it; the
  import itself checks the CPU and fails with a clear message otherwise.
- If the download is refused (no proxy on the login node), download the asset
  on another machine from
  `https://github.com/esmail-abdulfattah/sTiles/releases/latest` (asset
  `libstiles-linux-x86_64.zip`),
  copy it to Shaheen and point `STILES_LIB=/path/to/libstiles.so` to it in
  the shell set-up and in the job scripts instead of `STILES_CACHE_DIR`.
- Jobs run with `STILES_NO_DOWNLOAD=1`, so a missing cache fails fast instead
  of hanging on the network.

Note on tile modes: DALIA's sTiles solver defaults to `stiles_tile_mode =
"auto"`. The `sparse` mode is the only one free of the multithreaded
non-determinism of libstiles (`stiles_issue/ISSUE.md`), but from
`Version_2026.10.19` it fails on matrices that fit in a single tile
(n <= `stiles_tile_size`), which breaks the small models of the test suite.
With `auto` all 342 sTiles tests pass with the new release; the price is the
race (about 2 % of solves after a factorization were wrong at 8 cores on an
81k matrix in our measurements). Set `"stiles_tile_mode": "sparse"` in the
solver config of `run.py` to trade that for the single-tile limitation.

Speed of the releases, measured on the Spark (one process, 8 threads, full
GHCN case, nothing else running): `Version_2026.09.04` does one L-BFGS
iteration in 79 s, `Version_2026.10.19` in 129 s (sparse) or 139 s (auto),
with a different ordering and 7 % more factor entries. If the absolute
numbers matter more than using the latest release, pin the older one
(`pip install stiles==2026.9.4`, `STILES_RELEASE_TAG=Version_2026.09.04`
at the first import); the results are the same to 1e-7.

Quick functional check on the login node (small problem, 2 threads, ~1 min):

```bash
cd $DALIA/examples/gst_ghcn
python run.py --inputs small --solver stiles --threads 2 --max_iter 50
```

Expected at the end: `Norm (theta - theta_ref)` around 1e-9 and
`Norm (x - x_ref)` around 1e-7 against the Serinv reference shipped with the
example, 21 iterations.

## 3. Verify on a compute node (debug partition, ~5 min)

```bash
salloc --account=<project_id> --partition=debug --nodes=1 --time=00:20:00
# inside the allocation (the environment is inherited; re-source if not):
cd $DALIA/examples/gst_ghcn
export STILES_NO_DOWNLOAD=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun -n 1 -c 8 --cpu-bind=cores python run.py --inputs small --solver stiles --threads 8 --max_iter 50
srun -n 1 -c 8 --cpu-bind=cores python run.py --inputs small --solver serinv --max_iter 50
exit
```

Both must print the same hyperparameters and tiny norms against the reference.
If sTiles fails to import here but worked on the login node, the cache
directory is not under `/scratch`.

## 4. Data: download and prepare the inputs (login node)

```bash
mkdir -p /scratch/$USER/ghcn_raw && cd /scratch/$USER/ghcn_raw
curl -O https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2022.csv.gz      # 175 MB
curl -O https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt        # 11 MB

cd $DALIA/examples/gst_ghcn
python prepare_inputs.py --raw-dir /scratch/$USER/ghcn_raw --name full --mesh-nodes 1250
```

The first call parses the year file (~30 s) and caches it next to the raw
files; the full inputs take about a minute and 370 MB. Expected summary:
13,718 stations, 365 days, 4,310,481 observations, ns = 1250, nt = 365,
n_latent 456,250, 8 fixed effects.

## 5. Validate the full case before spending node-hours (debug, ~10 min)

One process, 32 threads, one L-BFGS iteration, to check loading, the symbolic
analysis, the memory and the time per evaluation on Genoa:

```bash
cd $DALIA/examples/gst_ghcn
sbatch --account=<project_id> --partition=debug --nodes=1 --ntasks=1 --cpus-per-task=32 \
       --hint=nomultithread --time=00:30:00 --job-name=ghcn_check --output=%x.%j.out \
       --wrap="export STILES_NO_DOWNLOAD=1 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32; \
               srun --cpu-bind=cores python run.py --inputs full --solver stiles --threads 32 --max_iter 1 --no-reference --tag check"
```

Look at the output file for:

- `sTiles symbolic analysis #1: n=456258, nnz(L)=595192293, ... s` (23 s on the Spark);
- `Iteration:  1 (took: ... )`: 9 function evaluations in sequence, 78 s on the
  Spark with 8 threads. Divide by 9 for the time per evaluation;
- the `RESULT ...` line and `outputs_full_stiles_r1_t32_check/summary.json`;
- `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS` for the peak memory
  (24.9 GB per process on the Spark; this is what bounds the ranks sweep).

If the time per evaluation is far above the Spark's 8.7 s at 8 threads, the
embedded MKL may be taking a slow path on AMD; try `--threads 8` for a direct
comparison before launching the sweeps.

## 6. Launch the sweeps (login node)

```bash
cd $DALIA/examples/gst_ghcn
./sweep_threads_shaheen3.sh <project_id>              # 9 jobs: 1 process x {1,2,4,8,16,32,48,96,192} threads
./sweep_ranks_shaheen3.sh   <project_id>              # 5 jobs: 1x192 2x96 6x32 9x21 12x16, sTiles
./sweep_ranks_shaheen3.sh   <project_id> full serinv  # 5 jobs: same configurations, Serinv baseline
squeue -u $USER
```

All jobs are single-node, exclusive on `workq`. Measured on Shaheen with
`Version_2026.10.19` (one process): 24.6 s per evaluation at 8 threads,
16.9 s at 32 threads, 65 s symbolic analysis, 25.4 GB peak memory. With
about 25 L-BFGS iterations plus the covariance phase, expect 1.5 to 2 h at
8 threads, 6 to 7 h at 1 thread, under 1 h from 32 threads on, and 20 to
40 min for the 9x21 and 12x16 runs. Total about 25 node-hours. Time limits
in the scripts: 12 h (1 and 2 threads), 6 h (up to 8 threads, 1 and 2
ranks), 4 h (threads) and 3 h (ranks) otherwise.

Each job writes `outputs_full_<solver>_r<ranks>_t<threads>_<jobid>/summary.json`
and a `RESULT ...` line in `ghcn_*.<jobid>.out`. Check progress with

```bash
grep -h "RESULT\|Traceback\|CANCELLED\|oom" ghcn_*.out
sacct -u $USER --format=JobName%22,Elapsed,MaxRSS,State -S today
```

If the 12x16 job dies with an out-of-memory kill, the sTiles analysis peaks
coincide on the 12 ranks; keep 9x21 as the top configuration. To re-run
only one configuration pass its parameters, e.g.
`./sweep_threads_shaheen3.sh <project_id> full 48`.

## 7. Collect and plot

On Shaheen:

```bash
cd $DALIA/examples/gst_ghcn
python collect_sweep.py                                        # sTiles -> stiles_fullsolve.csv
python collect_sweep.py --solver serinv --out serinv_fullsolve.csv
```

Copy both CSVs to the poster repository (through the multiplexed SSH
connection, no new OTP):

```bash
scp shaheen:/scratch/<user>/src/DALIA/examples/gst_ghcn/{stiles,serinv}_fullsolve.csv \
    ~/projects/dalia/repositories/KAUST_POSTER/figs_dalia/
cd ~/projects/dalia/repositories/KAUST_POSTER/figs_dalia
conda activate dalia_mpi && python make_figures.py             # layout C; or --stiles-layout A
cd .. && conda activate tex && tectonic dalia_poster.tex
```

Then replace the "Measured" bullet of the sTiles box in `dalia_poster.tex`
with the machine (Shaheen III, 2x AMD Genoa, 192 cores, 384 GB), the problem
size and the headline numbers.

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `could not download a solver; check the network, or set STILES_LIB` on a compute node | cache not under `/scratch`, or `STILES_CACHE_DIR` not exported in the job; redo step 2 and check the `export` lines |
| `Illegal instruction` at the first factorization | wrong native build; the default x86_64 asset needs AVX2, which Genoa has; make sure `STILES_VARIANT` is unset |
| `ValueError: This file contains pickled (object) data` | LFS pointers instead of data: `git lfs pull` in `$DALIA` |
| `error: account specification` at submission | missing or wrong `--account`; `sacctmgr show assoc user=$USER format=account` lists the valid ones |
| job killed, `sacct` shows `OUT_OF_MEMORY` | too many ranks per node for this problem; use 9x21 as the top configuration |
| `sTiles_chol failed (status -104)` on a tiny model | `stiles_tile_mode = "sparse"` with libstiles 2026.10.19 and n <= tile size; use the default `auto` mode or `Version_2026.09.04` |
| `theta must be positive` crash at iteration 2 | the observation precision lost its Gamma prior (log-scale optimisation); keep `run.py` as shipped |
| sTiles time per evaluation far above the Spark's 8.7 s at 8 threads | compare `--threads 8`; if still slow, the embedded MKL is on a slow path, report it with the `sacct` and output files |
