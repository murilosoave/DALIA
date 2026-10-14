import numpy as np
import pytest
import scipy.sparse as sps
from pydantic import ValidationError

from dalia.configs import submodels_config
from dalia.submodels import RW2SubModel

N = 12


def build(tmp_path, **extra):
    sps.save_npz(tmp_path / "a.npz", sps.identity(N, format="csr"))
    config = {
        "type": "rw2",
        "input_dir": str(tmp_path),
        "tau": 2.0,
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        **extra,
    }
    return RW2SubModel(config=submodels_config.parse_config(config))


def test_Q_prior_is_tau_R_plus_diagonal(tmp_path):
    submodel = build(tmp_path, diagonal=1e-3)
    D2 = np.diff(np.eye(N), n=2, axis=0)

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * D2.T @ D2 + 1e-3 * np.eye(N), rtol=1e-14, atol=1e-14)


def test_non_positive_diagonal_is_rejected(tmp_path):
    """Without the diagonal the precision is singular and cannot be factorized."""
    with pytest.raises(ValidationError):
        build(tmp_path, diagonal=0.0)


def test_given_constraints_come_after_the_identifiability_ones(tmp_path):
    """Identifiability rows first, sum(x) = 0 and sum((t - mean t) x) = 0 with
    t = 0..n-1, then the constraints of the config."""
    row = np.zeros((1, N))
    row[0, 0] = 1.0
    constraints = [{"type": "linear", "C": row.tolist(), "e": [0.5]}]
    t = np.arange(N, dtype=float)

    C, e = build(tmp_path, constraints=constraints).construct_constraints()

    np.testing.assert_allclose(
        np.asarray(C), np.vstack([np.ones((1, N)), t - t.mean(), row])
    )
    np.testing.assert_array_equal(np.asarray(e), np.array([0.0, 0.0, 0.5]))
