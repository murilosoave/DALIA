# gst_ghcn: Gaussian spatio-temporal model of GHCN-Daily temperatures

Real-data example reproducing the set-up of Lindgren, Bakka, Bolin, Krainski
and Rue (2024), "A diffusion-based spatio-temporal extension of Gaussian
Matérn fields", SORT 48(1), Section 5.2: the daily mean of TMIN and TMAX at the
stations of the Global Historical Climatology Network (GHCN-Daily, Menne et al.
2012) for one year, modelled with a spatio-temporal SPDE field on the sphere.

```
y(s, t) = mu + alpha * elevation(s) + b(lat(s), t) + u(s, t) + e
```

| Component | DALIA submodel | Size (full case) |
|---|---|---|
| `u(s, t)` daily field, one knot per day | `spatio_temporal`, `manifold = "sphere"` | 1,250 nodes x 365 days = 456,250 |
| `mu`, `alpha`, latitude-season basis `b` | `regression` | 8 fixed effects |
| `e` Gaussian noise | Gaussian likelihood, `prec_o` | 1 hyperparameter |

The latitude-season basis is the one of the paper: `{1, cos(2 pi (t-1)/365),
sin(2 pi (t-1)/365)} x {sin(lat), sin(lat)^2}`, zero at the equator. The
paper's slowly varying field `v(s, t)` (three spatial fields times the seasonal
basis) has no counterpart in DALIA and is omitted, and DALIA's field is
first-order in time (their models A/B), so the hyperparameters are not expected
to match the paper to the digit.

Priors follow the paper: PC priors with Pr(range_s < 600 km) = 0.01,
Pr(range_t < 1 day) = 0.01, Pr(sigma_st > 5 degC) = 0.01; Gaussian precision
0.01 on the fixed effects. The observation precision gets a weak Gamma(1, 0.01)
prior, because the Gamma prior is the one DALIA optimises on the log scale (with
the PC or Gaussian prior L-BFGS-B steps to a negative precision on this data).

## Data

GHCN-Daily is public (NOAA NCEI). Download, outside the repository:

```bash
mkdir -p /path/to/ghcn_raw && cd /path/to/ghcn_raw
curl -O https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2022.csv.gz        # 175 MB
curl -O https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt          # 11 MB
```

2022 has 13,718 stations with elevation and both TMIN and TMAX on at least
30 days, and 4.31 M station-days with both values and no quality flag (tenths of degC in the
raw file, converted to degC). Stations without elevation are dropped.

## Preparing inputs

`prepare_inputs.py` parses the raw files once (cached as `tmean_<year>.npz`
next to them, ~25 s), builds the spherical mesh and FEM matrices, the
space-time projection matrix and the design matrix, and writes an
`inputs_<name>/` directory in the layout of the other `gst_*` examples:

```
inputs_<name>/
  inputs_spatio_temporal/{c0,g1,g2,g3,m0,m1,m2,a}.npz
  inputs_spatial/{c0,g1,g2,a}.npz      # static field on the same mesh (not used by run.py, see below)
  inputs_regression/a.npz
  y.npy
  meta/{meta.json, stations.csv, mesh_loc.npy, mesh_tv.npy, obs_station.npy, obs_doy.npy}
```

Small case for local tests (300 random stations of the contiguous US, days 1 to
60, ~150 mesh nodes over the region plus a coarse mesh elsewhere, 13,440 latent
variables):

```bash
python prepare_inputs.py --raw-dir /path/to/ghcn_raw --name small \
    --region conus --n-stations 300 --days 1 60 --mesh-nodes 150
```

Full case of the paper (all stations, whole year, global 1,250-node mesh with a
median edge of 673 km, 456,250 latent variables, 4.31 M observations, 370 MB of inputs):

```bash
python prepare_inputs.py --raw-dir /path/to/ghcn_raw --name full --mesh-nodes 1250
```

Useful knobs: `--region conus|europe|australia|lat0,lat1,lon0,lon1`,
`--n-stations`, `--days FIRST LAST`, `--mesh-nodes` (a finer mesh is the way to
grow the problem: 2,000 nodes give ~730k latent variables), `--covariates
basic|full|auto` (the seasonal basis is only identifiable over a long window;
`auto` uses it for windows of at least 120 days), `--seed`.

Mesh and FEM conventions were checked against `fmesher::fm_fem` on a sphere
mesh: `c0` is the lumped mass matrix from the spherical triangle areas (it
sums to 4 pi), `g1` the stiffness matrix of the flat triangles, `g2 = g1 c0^-1 g1`,
`g3 = g2 c0^-1 g1`; all agree with fmesher to 1e-14. The temporal matrices are
those of `fm_fem(fm_mesh_1d(1:nt))`: `m0` the trapezoid-lumped mass, `m1` half
the boundary matrix, `m2` the second-difference stiffness. The projection
weights (barycentric coordinates of the radially projected station) differ from
`fmesher::fm_basis` by up to 1e-3.

Ranges are in radians on the unit sphere (1 rad = 6,371 km, 600 km = 0.094)
and in days; `run.py` prints them in km and days.

## Running

The complete Shaheen III procedure, from installing sTiles to the poster
figure, is in `SHAHEEN_BENCHMARK.md`.

```bash
# local, small case
python run.py --inputs small --solver stiles --threads 8
python run.py --inputs small --solver serinv

# options
python run.py --help
```

`--solver stiles|serinv|scipy|dense`, `--threads` and `--tile-size` for sTiles,
`--solver_min_p` for the distributed Serinv solver, `--max_iter`. Outputs
(`theta.npy`, `theta_internal.npy`, `x.npy`) go to `outputs_<inputs>_<solver>/`;
if `reference_outputs_<inputs>/` exists the run is compared against it.
`reference_outputs_small/` was produced with Serinv on the Spark; sTiles gives
the same optimum to all printed digits.

On Shaheen III the inputs must be prepared on a login node and live under
`/scratch`. Scripts:

- `../run_example_shaheen3_kaust.sh`: a single job (9 ranks x 21 threads).
- `sweep_threads_shaheen3.sh <project_id>`: the poster benchmark. One job per
  thread count (1, 2, 4, 8, 16, 32, 48, 96, 192), one process, one exclusive
  node each, measuring the wall time of the full inference (optimisation,
  hyperparameter covariance, marginal variances) of the full case.
- `sweep_ranks_shaheen3.sh <project_id> [inputs] [solver]`: ranks x threads =
  192 on one node (1x192, 2x96, 6x32, 9x21, 12x16), i.e. DALIA's
  function-evaluation parallelism around the shared-memory solver. One
  process of the full case peaks at ~25 GB during the sTiles symbolic
  analysis, which all ranks run at the same time, so 12 ranks (~300 GB) is the
  most that fits in a 384 GB node. With `serinv` as third argument the same
  configurations give the baseline.
- `collect_sweep.py`: gathers the `outputs_*/summary.json` of all runs into
  `stiles_fullsolve.csv`, the file the poster figure
  (`KAUST_POSTER/figs_dalia/make_figures.py`) reads.

Every run writes `outputs_<inputs>_<solver>_r<ranks>_t<threads>[_<tag>]/summary.json`
with the wall time, iteration count, sizes and hyperparameters, and prints a
one-line `RESULT ...` summary.

### Small-case results (Spark, 2026-10-25)

300 CONUS stations, days 1 to 60 of 2022, 17,410 observations, 224 mesh nodes:
sTiles (8 threads) and Serinv converge in 21 iterations to the same optimum,
41 s and 50 s respectively. Spatial range 5,300 km, temporal range 265 days,
sigma_st 14 degC, noise sd 2.75 degC. The very long temporal range and large
sigma_st show the spatio-temporal field absorbing the station climatology,
which the paper's slowly varying field `v(s, t)` would capture; with the full
year and the seasonal basis this is expected to be less pronounced.

### Full-case smoke test (Spark, 8 sTiles threads, 2 iterations)

n = 456,258, 4.31 M observations: symbolic analysis 23 s (nnz(L) = 5.95e8,
far from the libstiles limit met at ~2e9), 78 s per L-BFGS iteration (9
function evaluations in sequence, ~8.7 s each), hyperparameter covariance
285 s, 1.8 GB of host memory reported by DALIA. After two iterations the
hyperparameters are already in the expected region: spatial range ~1,000 km,
temporal range 3 days, sigma_st 5 degC, noise sd 1.9 degC. A converged run
needs ~20 to 30 iterations, i.e. 30 to 40 minutes on this machine with one
process; on Shaheen the 9 (or 18) function evaluations run on separate ranks.

## Known limitations

- The paper's `v(s, t)` term (spatially varying seasonal amplitude) cannot be
  expressed with the current submodels.
- Adding the static `spatial` submodel on the same mesh (inputs are generated
  in `inputs_spatial/`) fails at the first factorization with both sTiles and
  Serinv (`potrf info = 1` on the first block). Not investigated yet; a
  possible cause is the clash of the hyperparameter key `r_s` between the
  spatial and spatio-temporal submodels.
- The regional mesh (fine inside the region, coarse elsewhere) has a few thin
  triangles at the transition (minimum angle ~8 degrees in the small case);
  harmless for the smoke test, use a global mesh for results that matter.
- `prepare_inputs.py` needs only NumPy and SciPy (no pandas), so it runs in the
  DALIA environment on Shaheen.
