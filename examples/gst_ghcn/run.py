# Copyright 2024-2025 DALIA authors. All rights reserved.

"""GHCN-Daily global temperature: Gaussian spatio-temporal model with regression.

Inputs are produced by `prepare_inputs.py` into `inputs_<name>/` (see README.MD).

    python run.py --inputs small --solver stiles --threads 8
    srun python run.py --inputs full --solver stiles
"""

import argparse
import json
import os
import sys
import time

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dalia import comm_size
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import get_host, print_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EARTH_RADIUS_KM = 6371.0


def parse_args():
    p = argparse.ArgumentParser(description="DALIA GHCN example")
    p.add_argument("--inputs", default="small", help="name of the inputs_<name> directory")
    p.add_argument("--solver", default="stiles", choices=["stiles", "serinv", "scipy", "dense"])
    p.add_argument("--threads", type=int, default=None, help="sTiles threads per process (default: cores / processes per node)")
    p.add_argument("--tile-size", type=int, default=None, help="sTiles tile size (default: solver default)")
    p.add_argument("--solver_min_p", type=int, default=1, help="processes per Serinv solve (distributed solver if > 1)")
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--no-reference", action="store_true", help="do not compare against reference_outputs")
    p.add_argument("--tag", default="", help="suffix of the outputs directory (e.g. the SLURM job id)")
    return p.parse_args()


if __name__ == "__main__":
    print_msg("--- Example: GHCN-Daily temperature, Gaussian spatio-temporal model with regression ---")
    args = parse_args()
    inputs = f"{BASE_DIR}/inputs_{args.inputs}"
    with open(f"{inputs}/meta/meta.json") as fh:
        meta = json.load(fh)
    print_msg(f"inputs: {inputs}")
    print_msg(f"  {meta['n_stations']} stations, {meta['nt']} days, {meta['n_observations']} observations, "
              f"ns = {meta['ns']} ({meta['median_edge_km']:.0f} km median edge), fixed effects: {meta['fixed_effects']}")

    # --- Submodels
    # Priors follow Lindgren et al. (2024): Pr(range_s < 600 km) = 0.01,
    # Pr(range_t < 1 day) = 0.01, Pr(sigma > 5 deg C) = 0.01. Ranges are in
    # radians on the unit sphere (600 km = 0.094) and days; initial values on
    # the log scale.
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config({
            "type": "spatio_temporal",
            "input_dir": f"{inputs}/inputs_spatio_temporal",
            "spatial_domain_dimension": 2,
            "manifold": "sphere",
            "r_s": float(np.log(1000.0 / EARTH_RADIUS_KM)),
            "r_t": float(np.log(3.0)),
            "sigma_st": float(np.log(5.0)),
            "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 600.0 / EARTH_RADIUS_KM},
            "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 1.0},
            "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 5.0},
        })
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config({
            "type": "regression",
            "input_dir": f"{inputs}/inputs_regression",
            "n_fixed_effects": meta["n_fixed_effects"],
            "fixed_effects_prior_precision": 0.01,
        })
    )
    submodels = [spatio_temporal, regression]

    # --- Likelihood: Gaussian noise. A weak Gamma prior on the precision: the
    # Gamma prior is the one that optimises prec_o on the log scale (the PC and
    # Gaussian priors use the identity transform and let L-BFGS-B step to a
    # negative precision).
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 0.01},
    }

    model = Model(
        submodels=submodels,
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # --- DALIA
    solver = {"type": args.solver, "min_processes": args.solver_min_p}
    if args.solver == "stiles":
        if args.threads is not None:
            solver["stiles_threads"] = args.threads
        if args.tile_size is not None:
            solver["stiles_tile_size"] = args.tile_size
    dalia_dict = {
        "solver": solver,
        "minimize": {"max_iter": args.max_iter, "gtol": 1e-3, "disp": True},
        "f_reduction_tol": 1e-4,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(model=model, config=dalia_config.parse_config(dalia_dict))

    t0 = time.perf_counter()
    results = dalia.run()
    elapsed = time.perf_counter() - t0

    # --- Results
    theta = get_host(results["theta"])
    x = get_host(results["x"])
    print_msg("\n--- Results ---")
    print_msg(f"wall time DALIA.run(): {elapsed:.1f} s")
    keys = list(model.theta_keys)
    print_msg("theta (interpretable scale):")
    for key, val in zip(keys, theta):
        if key == "r_s":
            extra = f"spatial range {np.exp(val) * EARTH_RADIUS_KM:.0f} km"
        elif key == "r_t":
            extra = f"temporal range {np.exp(val):.2f} days"
        elif key in ("sigma_st", "sigma_e"):
            extra = f"sd {np.exp(val):.2f} deg C"
        elif key == "prec_o":
            prec = val if val > 0 else np.exp(val)
            extra = f"noise sd {1.0 / np.sqrt(prec):.2f} deg C"
        else:
            extra = ""
        print_msg(f"  {key:10s} {val: .4f}   {extra}")
    n_fixed = meta["n_fixed_effects"]
    fixed = x[model.submodels[0].n_latent_parameters: model.submodels[0].n_latent_parameters + n_fixed]
    print_msg("fixed effects:")
    for name, val in zip(meta["fixed_effects"], fixed):
        print_msg(f"  {name:16s} {val: .4f}")

    threads = args.threads if args.solver == "stiles" else 0
    out_dir = f"{BASE_DIR}/outputs_{args.inputs}_{args.solver}_r{comm_size}_t{threads}"
    if args.tag:
        out_dir += f"_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/theta.npy", theta)
    np.save(f"{out_dir}/theta_internal.npy", get_host(results["theta_internal"]))
    np.save(f"{out_dir}/x.npy", x)
    summary = {
        "case": f"ghcn_{args.inputs}",
        "inputs": args.inputs,
        "solver": args.solver,
        "ranks": int(comm_size),
        "threads": threads,
        "tile_size": args.tile_size,
        "n": int(model.n_latent_parameters),
        "n_observations": meta["n_observations"],
        "wall_time_s": elapsed,
        "iterations": int(results.get("optimization_iterations", -1)),
        "theta": [float(v) for v in theta],
        "theta_keys": keys,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print_msg(f"outputs written to {out_dir}")
    print_msg(f"RESULT case={summary['case']} ranks={comm_size} threads={threads} "
              f"iterations={summary['iterations']} wall_time_s={elapsed:.1f}")

    if not args.no_reference:
        ref_dir = f"{BASE_DIR}/reference_outputs_{args.inputs}"
        if os.path.isdir(ref_dir):
            print_msg("\n--- Comparisons against reference outputs ---")
            theta_ref = np.load(f"{ref_dir}/theta_ref.npy")
            x_ref = np.load(f"{ref_dir}/x_ref.npy")
            print_msg(f"Norm (theta - theta_ref): {np.linalg.norm(theta - theta_ref):.4e}")
            print_msg(f"Norm (x - x_ref):         {np.linalg.norm(x - x_ref):.4e}")
        else:
            print_msg(f"\n(no reference outputs in {ref_dir})")

    print_msg("\n--- Finished ---")
