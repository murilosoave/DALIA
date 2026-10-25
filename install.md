# How to install DALIA on one of the default clusters or your personal machine
DALIA has been developed on several (super)computing infrastructures. Since the beginning of the project, our goal has been to provide a seamless experience for users regardless of the underlying hardware and software stack.

Because full generalization is not always possible, we provide a simplified procedure for multiple supercomputing infrastructures (with different hardware) to make installing DALIA on a new cluster as easy as possible.

## Purpose
Base `conda` environments for DALIA are provided in the `dalia/envs` directory. These environments contain all the Python packages on which DALIA relies. Hardware-specific packages (e.g., `cupy` for GPU compute, `mpi4py` for multiprocessing) are not included; they can be installed as extensions of the base environment. We provide base environments for both `x86` and `aarch64` architectures.

These environments are used in the CI/CD pipelines for the respective clusters and are also available for users to quickly set up DALIA on these clusters or any other machine.

## Supported Clusters
We currently use four different clusters for the testing and development of DALIA:

| Organization | Cluster | Arch                              | Description                                                    | Memory                     | Nodes                        |
| ------------ | ------- | --------------------------------- | -------------------------------------------------------------- | -------------------------- | ---------------------------- |
| [FAU](https://doc.nhr.fau.de/)          | [Fritz](https://doc.nhr.fau.de/clusters/fritz/)   | x86 (Sapphire Rapids)             | 2x Intel Xeon Platinum 8470 per node<br>(2x52-cores @ 2.0 GHz) | Up to 2TB DDR5             | 64x 8470, 992x 8360Y         |
| [FAU](https://doc.nhr.fau.de/)          | [Alex](https://doc.nhr.fau.de/clusters/alex/)    | x86 (AMD EPYC 7713)<br>Ampere<br> | 8x Nvidia A100 per node<br>(2x 64-cores CPU + 8 GPUs)          | Up to 80GB HBM2            | 18x A100 80GB, 20x A100 40GB |
| [CSCS](https://www.cscs.ch/)         | [Daint](https://docs.cscs.ch/clusters/daint/)   | ARM (Grace)<br>Hopper             | 4x Nvidia GH200 per node<br>(72-cores CPU + 1 GPU)             | 128GB LPDDR5X<br>96GB HBM3 | 1022                         |
| [KAUST](https://docs.hpc.kaust.edu.sa/)        | [Shaheen III](https://docs.hpc.kaust.edu.sa/quickstart/shaheen3.html) | x86 (AMD Genoa)<br>ARM (Grace) + Hopper | 2x AMD Genoa per node (192 cores)<br>GPU partition: Nvidia GH200 | 384GB DDR5 (CPU nodes) | 4608 CPU nodes<br>2800 GH200 |


## Environments Configuration Matrix
DALIA is designed to work across a wide variety of hardware and software stacks (e.g., GPU acceleration, distributed memory, vendor-specific libraries). The table below summarizes the supported configurations and the corresponding pre-configured `conda` environments.

|              | No Comm              | Host MPI           | GPU-Aware MPI      | xCCL               |
| :----------- | :------------------- | :----------------- | :----------------- | :----------------- |
| CPU (x86)    | *dalia_base_fritz* | *dalia_hmpi_fritz* | NA                 | NA                 |
| GPU (NVIDIA) | *dalia_base_alex*  | NA  | NA  | *dalia_xccl_alex*  |
| GPU (NVIDIA) | *dalia_base_daint* | NA | NA | *dalia_xccl_daint* |
| CPU (x86)    | *dalia_base_shaheen* | *dalia_hmpi_shaheen* | NA | NA |
| CPU (ARM)    | x                    | x                  | x                  | x                  |
| GPU (AMD)    | x                    | x                  | x                  | x                  |

The `dalia_base` environment contains all the dependencies needed to run DALIA on a single node without any communication library (e.g., MPI, xCCL) and without GPU support. This environment includes only hardware-independent Python dependencies. In `dalia/scripts/` we provide interactive installers that can create these `dalia_base` environments for you and, when applicable, extend them to support multi-node communication with MPI or xCCL and/or GPU acceleration.

**Notes:** 
- On the Daint and Alex clusters, CuPy is installed using wheels that include NCCL. Therefore, NCCL is available whenever CuPy is installed. For this reason, no standalone GPU-aware MPI environments are provided.
- On Daint, the `dalia_base_daint` environment allows one to run on the ARM CPU. However, we do not provide any conda environment specifically targeted for a general ARM CPU-cluster.
- On Shaheen III, no installer script is provided yet; the environments are created manually following the step-by-step instructions below. `mpi4py` is built against Cray MPICH.

# Detailed Instructions
## On Fritz@FAU
### a) Installation
1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `fritz_fau_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/fritz_fau_utils.sh
    ```
3. Load the required environment modules:
    ```bash
    fritz_load_modules
    ```
4. Create the conda environment:
    ```bash
    fritz_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `fritz_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```bash
    fritz_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env="desired_environment"` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `fritz_fau_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/fritz_fau_utils.sh
    ```
2. Load the required environment modules:
    ```bash
    fritz_load_modules
    ```
3. Activate the conda environment:
    ```bash
    fritz_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc -N 1 --time=00:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_fritz_fau.sh` script for an example on how to submit a job on Fritz.

## On Alex@FAU
### a) Installation

1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `alex_fau_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/alex_fau_utils.sh
    ```
3. Load the required environment modules:
    ```bash
    alex_load_modules
    ```
4. Create the conda environment:
    ```bash
    alex_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `alex_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```bash
    alex_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `alex_fau_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/alex_fau_utils.sh
    ```
2. Load the required environment modules:
    ```bash
    alex_load_modules
    ```
3. Activate the conda environment:
    ```bash
    alex_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc --gres=gpu:a100:1 --time=0:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_alex_fau.sh` script for an example on how to submit a job on Alex.


## On Daint@CSCS
### Preamble

In order to successfully install DALIA or even clone the repository on Daint, you will need to install `git-lfs` (Git Large File Storage) in your user space (see "Notes on `git-lfs`" for details). We provide a gist utility script alongside the following installation instructions:

```bash
mv /my/install/path
git clone https://gist.github.com/vincent-maillou/d2d38937f7aafbf0cee98c65cf5cfbca # Get the installation script
cd d2d38937f7aafbf0cee98c65cf5cfbca/ # Move into the script directory

chmod u+x daint_install_git_lfs.sh # Render the script executable
./daint_install_git_lfs.sh # Run the installation script
export PATH="$HOME/.local/bin:$PATH" # Add git-lfs to your PATH

which git-lfs # Verify the installation
```

Git-lfs will now be installed and available in your user space. Its path have been added to your `.bashrc` file, so it will be available in future shell sessions.


### a) Installation
1. Clone the repositories to your workspace:    
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Source the `daint_cscs_utils.sh` script to access the install utilities:
    ```bash
    cd DALIA/
    source scripts/daint_cscs_utils.sh
    ```

3. Use the `daint_install_conda` utility to install Miniconda in your user space (if not already installed):
    ```bash
    daint_install_conda --yes
    ```
    **Notes:**
    - This step is only required if you do not already have `conda` installed.
    - The `--yes` option automatically confirms the installation prompts.
    - Additional information about the installer options can be found by running `daint_install_conda --help`.

4. Source the installed conda environment:
    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
    **Notes:**
    - This is needed only once; in subsequent shell sessions conda will be initialized automatically through your `.bashrc` file.

5. Install the programming environment (`uenv`):
    ```bash
    daint_install_uenv
    ```

6. Start the programming environment:
    ```bash
    daint_start_uenv
    ```
    **Notes:**
    - This script assumes that you have successfully installed the programming environment in the previous step. If a programming environment is already active, it will be stopped and replaced with a new one.

7. Source the `daint_cscs_utils.sh` script again to access the install utilities:
    ```bash
    source scripts/daint_cscs_utils.sh
    ```
    **Notes:**
    - This is needed because starting the programming environment spawns a new shell session in which previously sourced scripts are no longer available.

8. Load the required environment modules:
    ```bash
    daint_load_modules
    ```

9. Create the conda environment:
    ```bash
    daint_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```

10. Activate the conda environment:
    ```bash
    daint_activate_conda_env
    ```
    Note: This function tries to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `daint_cscs_utils.sh` script:
    ```bash
    source /path/to/DALIA/scripts/daint_cscs_utils.sh
    ```
2. Start the programming environment:
    ```bash
    daint_start_uenv
    ```
3. Source the `daint_cscs_utils.sh` script again (`uenv` is starting a new shell) to access the install utilities:
    ```bash
    source /path/to/DALIA/scripts/daint_cscs_utils.sh
    ```
4. Load the required environment modules:
    ```bash
    daint_load_modules
    ```
5. Activate the conda environment:
    ```bash
    daint_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. Directly from the front node (after activating the correct conda environment) you can run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_daint_cscs.sh` script for an example on how to submit a job on Daint.

## On Shaheen III@KAUST
### Preamble
Shaheen III is an HPE Cray EX system. A few things differ from the FAU and CSCS clusters and shape the instructions below:

- **Accounts:** every job must be charged to a project (`#SBATCH --account=<project_id>`). Your PI adds you to the project; the quickstart documentation does not cover account creation. Use `kpq` to check the project balance and `groupies` to list project members.
- **Filesystems:** `/home` is **not mounted on compute nodes** (it is redirected to `/scratch/$USER`) and `/project` is **read-only** on compute nodes. Everything DALIA reads and writes at runtime (code, data, outputs, conda environments) must live under `/scratch/$USER`. Software goes in the high-IOPS tier `$MY_SW` (`/scratch/$USER/iops/sw`), which is exempt from the 60-day scratch purge. The rest of `/scratch` is purged after 60 days.
- **Inodes:** `/scratch` is limited to 1 million files per user. Conda environments are file-heavy, so always run `conda clean --all` after creating one.
- **Internet:** assume compute nodes have no internet access. Do all cloning, downloads and conda/pip installs from the login nodes.
- **MPI:** the system MPI is Cray MPICH over Slingshot. `mpi4py` must be built against it with the `cc` compiler wrapper (see step 7). Do **not** `conda install mpi4py`, which pulls in its own MPICH and will not use the interconnect.
- **git-lfs:** not provided by the system. Install it in your user space (see "Notes on `git-lfs`" > Shaheen III) before cloning, or run `git lfs pull` afterwards. Without it the LFS-tracked example data are left as text pointers and loading them fails with a misleading `pickled (object) data` error.

Login (the hostname load-balances across 5 identical login nodes):
```bash
ssh -X <username>@shaheen.hpc.kaust.edu.sa
```

### a) Installation
1. Clone the repositories into `/scratch`:
    ```bash
    mkdir -p /scratch/$USER/src && cd /scratch/$USER/src
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Install Miniconda in the high-IOPS tier (one-off). Shaheen keeps a local copy of the installer, so no download is needed:
    ```bash
    mkdir -p $MY_SW && cd $MY_SW
    bash /sw/sources/miniconda/conda24.1.2-python3.12.1/Miniconda3-latest-Linux-x86_64.sh -b -s -p $MY_SW/miniconda3-amd64 -u
    source $MY_SW/miniconda3-amd64/bin/activate
    conda install -y -c conda-forge mamba==1.5.8
    ```
    **Notes:**
    - Do **not** let the installer modify your `.bashrc` (the `-b` flag prevents this). Activate conda explicitly in every shell and job script with `source $MY_SW/miniconda3-amd64/bin/activate`.
    - The system `python` and `cray-python` modules cannot be used: DALIA requires Python >= 3.13, which the conda environment provides.

3. Switch to the GNU programming environment. The default is `PrgEnv-cray`; the GNU environment matches the GCC-built binaries shipped by conda-forge and is needed to build `mpi4py`:
    ```bash
    module switch PrgEnv-cray PrgEnv-gnu
    ```

4. Create the base conda environment. The CPU partition is x86 (AMD Genoa), so use the `x86` environment file and install it under `$MY_SW`:
    ```bash
    mamba env create -f /scratch/$USER/src/DALIA/envs/dalia_base_x86.yml -p $MY_SW/envs/dalia_base_shaheen
    conda clean --all -y
    ```

5. Activate the environment:
    ```bash
    conda activate $MY_SW/envs/dalia_base_shaheen
    ```

6. Install DALIA (and optionally serinv) in editable mode:
    ```bash
    cd /scratch/$USER/src/DALIA && python -m pip install --no-deps --editable .
    cd /scratch/$USER/src/serinv && python -m pip install --no-deps --editable . # Optional
    ```

7. (Optional, recommended) Add host-MPI support. Clone the base environment and build `mpi4py` from source against Cray MPICH via the compiler wrapper:
    ```bash
    conda deactivate
    conda create -p $MY_SW/envs/dalia_hmpi_shaheen --clone $MY_SW/envs/dalia_base_shaheen -y
    conda activate $MY_SW/envs/dalia_hmpi_shaheen
    MPICC=cc python -m pip install --no-cache-dir --no-binary mpi4py mpi4py
    conda clean --all -y
    ```
    Verify the build picked up Cray MPICH:
    ```bash
    python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
    ```
    The output should mention `MPI VERSION    : CRAY MPICH`.

### b) Usage
There is currently no `shaheen3_kaust_utils.sh` helper script; the equivalent of `*_load_modules` and `*_activate_conda_env` is the following, which must be run in every new shell session and at the top of every job script:
```bash
module switch PrgEnv-cray PrgEnv-gnu
source $MY_SW/miniconda3-amd64/bin/activate $MY_SW/envs/dalia_hmpi_shaheen  # or dalia_base_shaheen
```

Partitions relevant to DALIA (limits are per user):

| Partition | Tenancy   | Nodes | Max walltime        | Use                              |
| --------- | --------- | ----- | ------------------- | -------------------------------- |
| `debug`   | exclusive | 4     | 30 min (max 4 h)    | test suite, short runs           |
| `workq`   | exclusive | 4608  | 24 h                | production                       |
| `shared`  | shared    | 16    | 24 h                | small single-process runs        |

Each node has 2x AMD Genoa (192 physical cores, 384 GB). `workq` and `debug` allocate whole nodes and charge 192 core-hours per node-hour regardless of usage, so size the MPI ranks x threads to fill the node. Example job script (save it under `/scratch/$USER` and submit from there):
```bash
#!/bin/bash -l
#SBATCH --job-name="dalia_shaheen3"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=<project_id>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

module switch PrgEnv-cray PrgEnv-gnu
source $MY_SW/miniconda3-amd64/bin/activate $MY_SW/envs/dalia_hmpi_shaheen

source /scratch/$USER/src/DALIA/scripts/dalia_job_utils.sh && dalia_set_perfenv && dalia_print_job_config
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /scratch/$USER/src/DALIA/examples
srun --hint=nomultithread python ./gr/run.py --max_iter 100
```
Submit with `sbatch <script>.slurm`. Note that `ARRAY_MODULE=numpy` (set by `dalia_set_perfenv`) is the correct choice on the CPU partition.

A ready-made version of this script is `examples/run_example_shaheen3_kaust.sh` (run it from the `examples/` directory, after filling in the account), and `examples/gst_ghcn/sweep_threads_shaheen3.sh` / `sweep_ranks_shaheen3.sh` submit the sTiles thread and ranks sweeps on the GHCN-Daily example.

### c) Verify Installation
1. Run the test suite in an interactive allocation on the `debug` partition:
    ```bash
    salloc --account=<project_id> --partition=debug --nodes=1 --time=00:30:00
    module switch PrgEnv-cray PrgEnv-gnu
    source $MY_SW/miniconda3-amd64/bin/activate $MY_SW/envs/dalia_hmpi_shaheen
    cd /scratch/$USER/src/DALIA/tests && ./runner.sh
    ```
    **Note:** `runner.sh` uses paths relative to the `tests/` directory, so it must be launched from there. Running it from elsewhere fails with `ERROR: file or directory not found: unit/`.
2. Submit the example job script above with one of the examples uncommented.

### d) GPU partition (GH200)
Shaheen III also has 2,800 NVIDIA GH200 Grace Hopper superchips. These are **aarch64** (Grace) hosts, so the x86 environment above cannot be used on them. At the time of writing the public KAUST documentation does not describe the GPU partition (partition name, CUDA modules), so the following is a starting point to be confirmed with `module avail` and `sinfo` once you have access:

- Install a second, aarch64 Miniconda (`Miniconda3-latest-Linux-aarch64.sh`) in `$MY_SW` and create the environment from `envs/dalia_base_aarch64.yml`.
- Install CuPy with the CUDA 12 wheel so that NCCL is bundled: `pip install cupy-cuda12x==13.2.0`. Remember that NCCL is lazily imported (`from cupy.cuda import nccl`).
- Build `mpi4py` with `MPICC=cc` as above, with the NVIDIA accelerator target loaded (`module load craype-accel-nvidia90 cudatoolkit`) so that GPU-aware MPI is available. In the job script set `MPICH_GPU_SUPPORT_ENABLED=1`, `ARRAY_MODULE=cupy` and, if using NCCL, `USE_NCCL=1`.
- Mirror the Daint job scripts (`examples/run_example_daint_cscs.sh`) for the `srun` layout: one rank per GPU.

## On a Personal Machine
### a) Installation
Given a working installation of `git` and `conda`, you can install DALIA on your personal machine as follows:

1. Clone the repositories to your workspace:
    ```bash
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Based on your micro-architecture (x86 or aarch64), create the conda environment using the provided installer scripts:
- For x86 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_x86.yml
    ```
- For aarch64 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_aarch64.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate env_name
    ```

4. Install DALIA in editable mode:
    ```bash
    cd path/to/DALIA
    pip install -e .
    ```

5. Install Serinv (optional, recommended for ST-modeling) in editable mode:
    ```bash
    cd path/to/serinv
    pip install -e .
    ```

### b) Verify Installation
They are currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, for example: `python /path/to/DALIA/examples/gr/run.py`

## Notes on `git-lfs`

### Overview
DALIA uses `git-lfs` (Git Large File Storage) to manage large files such as example datasets. Files tracked by `git-lfs` are stored as pointers in the repository and downloaded on-demand rather than during the initial clone. This is particularly beneficial for large binary files (images, audio files, datasets) that don't compress well.

### Availability by Cluster

#### Fritz and Alex (FAU)
- `git-lfs` is available by default
- Location: `/usr/bin/git-lfs`
- No additional installation required

#### Daint (CSCS)
- `git-lfs` is **not** provided by the system
- Must be installed manually in your user space
- Installation script available at: https://gist.github.com/vincent-maillou/d2d38937f7aafbf0cee98c65cf5cfbca

#### Shaheen III (KAUST)
- `git-lfs` is **not** provided by the system (login nodes can still `git clone`, but LFS-tracked files are left as text pointers)
- Symptom: tests or examples that load `examples/**/*.npz` fail with `ValueError: This file contains pickled (object) data`
- Install the amd64 binary in your user space from a login node, then pull the data:
    ```bash
    mkdir -p ~/.local/bin && cd $MY_SW
    curl -LO https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz
    tar xzf git-lfs-linux-amd64-v3.6.1.tar.gz && cp git-lfs-3.6.1/git-lfs ~/.local/bin/
    export PATH=$HOME/.local/bin:$PATH   # add to ~/.bashrc as well
    cd /scratch/$USER/src/DALIA && git lfs install && git lfs pull
    ```

### Important Note on Repository Cloning
Without `git-lfs` installed on Daint, `git clone` will fail because it automatically triggers `git-lfs checkout` during the checkout process. This is why the Daint installation instructions include a `git-lfs` installation step in the preamble (see section "On Daint@CSCS > Preamble" for detailed instructions).

## Other Informations

- After installing packages with conda, it is recommended to run `conda clean --all` to free up disk space.
- These installation procedures and `conda` environments have been tested on Linux systems based on both `x86` and `aarch64` architectures.
- If the `sqlite` module does not work properly, forcing the following version might help:
    ```bash
    conda install conda-forge::sqlite=3.45.3
    ```
- CuPy switched NCCL to lazy import, which means NCCL must be imported explicitly with `from cupy.cuda import nccl` before it is available in `cupy.cuda.nccl`.
