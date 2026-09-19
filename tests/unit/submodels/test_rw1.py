# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest

from dalia.configs import submodels_config
from dalia.configs.submodels_config import RW1SubModelConfig
from dalia.submodels import RW1SubModel
from dalia.utils import get_host
from tests.constraints_utils import N_LATENT_AR1, make_dataset, rw1_dict


@pytest.fixture
def rw1(tmp_path):
    root = make_dataset(tmp_path)
    return RW1SubModel(config=submodels_config.parse_config(rw1_dict(root, tau=2.5)))


def test_config_parses_type_and_hyperparameters(tmp_path):
    root = make_dataset(tmp_path)
    cfg = submodels_config.parse_config(rw1_dict(root, tau=2.5))
    assert isinstance(cfg, RW1SubModelConfig)
    theta, keys = cfg.read_hyperparameters()
    assert keys == ["tau"]
    np.testing.assert_array_equal(get_host(theta), [2.5])


def test_is_intrinsic_with_constant_null_space(rw1):
    assert rw1.intrinsic is True
    N = rw1.null_space()
    assert N.shape == (N_LATENT_AR1, 1)
    np.testing.assert_array_equal(N[:, 0], 1.0)
    c = rw1.null_space_constraint
    assert c.k == 1
    np.testing.assert_array_equal(get_host(c.A.toarray()), np.ones((1, N_LATENT_AR1)))


def test_precision_is_tau_times_first_difference_laplacian(rw1):
    n = N_LATENT_AR1
    Q = get_host(rw1.construct_Q_prior(tau=2.5).toarray())
    D = np.diff(np.eye(n), axis=0)
    np.testing.assert_allclose(Q, 2.5 * D.T @ D)
    np.testing.assert_allclose(Q @ np.ones(n), 0.0, atol=1e-14)
    # tridiagonal with the diagonal stored (needed by the model's solver copy)
    coo = rw1.construct_Q_prior(tau=1.0)
    assert coo.nnz == 3 * n - 2
    assert np.all(np.abs(get_host(coo.row) - get_host(coo.col)) <= 1)


def test_generalized_logdet_matches_nonzero_eigenvalues(rw1):
    n = N_LATENT_AR1
    for tau in (0.3, 1.0, 7.0):
        Q = get_host(rw1.construct_Q_prior(tau=tau).toarray())
        eig = np.linalg.eigvalsh(Q)
        eig = eig[eig > 1e-10]
        assert eig.size == n - 1
        np.testing.assert_allclose(
            rw1.logdet_Q_prior_generalized(tau=tau), np.sum(np.log(eig)), rtol=1e-12
        )


def test_requires_at_least_two_latent_parameters(tmp_path):
    from scipy.sparse import csr_matrix, save_npz

    (tmp_path / "inputs_tiny").mkdir()
    save_npz(tmp_path / "inputs_tiny" / "a.npz", csr_matrix(np.ones((3, 1))))
    cfg = submodels_config.parse_config(
        {"type": "rw1", "input_dir": str(tmp_path / "inputs_tiny"), "tau": 1.0,
         "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0}}
    )
    with pytest.raises(ValueError, match="at least 2"):
        RW1SubModel(config=cfg)


def test_str_mentions_type_and_rank_deficiency(rw1):
    text = str(rw1)
    assert "RW1" in text and "Rank deficiency" in text
