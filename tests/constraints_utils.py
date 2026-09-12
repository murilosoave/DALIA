# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Small synthetic AR1 + intercept datasets written to disk in the layout SubModel/Model expect."""

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz


N_LATENT_AR1 = 12
N_OBS = 36


def make_dataset(
    root: Path, n: int = N_LATENT_AR1, n_obs: int = N_OBS, likelihood: str = "gaussian", seed: int = 0
) -> Path:
    """Write inputs_ar1/a.npz (n_obs x n), inputs_regression/a.npz (n_obs x 1) and y.npy under root."""
    rng = np.random.default_rng(seed)
    (root / "inputs_ar1").mkdir(parents=True, exist_ok=True)
    (root / "inputs_regression").mkdir(parents=True, exist_ok=True)

    idx = rng.integers(0, n, size=n_obs)
    a_ar1 = csr_matrix((np.ones(n_obs), (np.arange(n_obs), idx)), shape=(n_obs, n))
    a_reg = csr_matrix(np.ones((n_obs, 1)))
    save_npz(root / "inputs_ar1" / "a.npz", a_ar1)
    save_npz(root / "inputs_regression" / "a.npz", a_reg)

    x_true = rng.normal(size=n)
    x_true -= x_true.mean()
    eta = a_ar1 @ x_true + 0.3
    if likelihood == "gaussian":
        y = eta + rng.normal(scale=0.3, size=n_obs)
    elif likelihood == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        raise ValueError(likelihood)
    np.save(root / "y.npy", y)
    return root


def ar1_dict(root: Path, **extra) -> dict:
    return {
        "type": "ar1",
        "input_dir": str(root / "inputs_ar1"),
        "phi": 0.5,
        "ph_phi": {"type": "beta", "alpha": 2.0, "beta": 2.0},
        "tau": 1.0,
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        **extra,
    }


def regression_dict(root: Path) -> dict:
    return {
        "type": "regression",
        "input_dir": str(root / "inputs_regression"),
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }


def likelihood_dict(likelihood: str, root: Path | None = None) -> dict:
    if likelihood == "gaussian":
        return {
            "type": "gaussian",
            "prec_o": 10.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 0.1},
        }
    if likelihood == "poisson":
        # input_dir is required: PoissonLikelihood looks for an optional e.npy there.
        return {"type": "poisson", "input_dir": str(root)}
    raise ValueError(likelihood)
