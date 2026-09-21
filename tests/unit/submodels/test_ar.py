# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Tests of the stationary AR(p) submodel.

The references in this file are independent of `dalia.submodels.ar`:

- `_legacy_ar1_Q` and `_legacy_ar2_Q` are the closed-form matrices of the
  specialized AR1 / AR2 submodels that the general AR(p) submodel replaced.
- `_lyapunov_Q` inverts the stationary covariance obtained from the companion
  state Lyapunov equation and an independent pacf -> phi recursion.
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

from dalia import xp
from dalia.configs import submodels_config
from dalia.configs.submodels_config import ARSubModelConfig
from dalia.submodels import ARSubModel
from dalia.utils.gpu_utils import get_host

RTOL = 1e-10
ATOL = 1e-10

BETA_PM1 = {"type": "beta", "alpha": 2.0, "beta": 2.0, "support": [-1.0, 1.0]}
GAMMA = {"type": "gamma", "alpha": 2.0, "beta": 1.0}

PACF_VALUES = [-0.9, -0.35, 0.0, 0.2, 0.75]
TAUS = [0.2, 1.0, 37.5]

PACFS_HIGHER_ORDER = [
    [0.6, -0.4, 0.3],
    [-0.5, 0.2, -0.7],
    [0.0, 0.0, 0.8],
    [0.3, 0.0, -0.2, 0.5],
    [0.9, -0.8, 0.7, -0.6, 0.5],
    [-0.2, 0.0, 0.0, 0.0, 0.0, 0.4],
]


# --- Helpers ------------------------------------------------------------------
def _dense(Q) -> np.ndarray:
    return np.asarray(get_host(Q.toarray()))


def _write_inputs(path, n):
    scipy.sparse.save_npz(path / "a.npz", scipy.sparse.eye(n, format="csc"))
    return str(path)


def _ar_dict(input_dir, pacf, tau=1.0):
    return {
        "type": "ar",
        "input_dir": input_dir,
        "order": len(pacf),
        "pacf": list(pacf),
        "ph_pacf": [dict(BETA_PM1) for _ in pacf],
        "tau": tau,
        "ph_tau": dict(GAMMA),
    }


def _make_ar(tmp_path, n, pacf, tau=1.0) -> ARSubModel:
    config = submodels_config.parse_config(
        _ar_dict(_write_inputs(tmp_path, n), pacf, tau)
    )
    return ARSubModel(config=config)


def _Q(tmp_path, n, pacf, tau) -> np.ndarray:
    """Dense prior precision of an ARSubModel with n latent parameters."""
    ar = _make_ar(tmp_path, n, [0.0] * len(pacf))
    return _dense(ar.construct_Q_prior(**_pacf_kwargs(pacf, tau)))


def _pacf_kwargs(pacf, tau):
    kwargs = {f"pacf{k}": v for k, v in enumerate(pacf, start=1)}
    kwargs["tau"] = tau
    return kwargs


# --- Independent references ---------------------------------------------------
def _legacy_ar1_Q(n, phi, tau):
    """Closed form of the former AR1SubModel.construct_Q_prior."""
    s2 = 1 / tau
    denom = s2 * (1 - phi**2)

    diag = [(1 + phi**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    off_diag = [-phi / denom] * (n - 1)

    return scipy.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1]).toarray()


def _legacy_ar2_Q(n, pacf1, pacf2, tau):
    """Closed form of the former AR2SubModel.construct_Q_prior."""
    phi2 = pacf2
    phi1 = pacf1 * (1 - pacf2)

    s2 = 1 / tau
    denom = s2 * (1 + phi2) * ((1 - phi2) ** 2 - phi1**2) / (1 - phi2)

    diag = [(1 + phi1**2 + phi2**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    diag[1] = diag[-2] = (1 + phi1**2) / denom

    off_diag_1 = [-phi1 * (1 - phi2) / denom] * (n - 1)
    off_diag_1[0] = off_diag_1[-1] = -phi1 / denom

    off_diag_2 = [-phi2 / denom] * (n - 2)

    return scipy.sparse.diags(
        [off_diag_2, off_diag_1, diag, off_diag_1, off_diag_2],
        [-2, -1, 0, 1, 2],
    ).toarray()


def _reference_phi(pacf):
    """Scalar Durbin-Levinson recursion, written independently of the library."""
    phi = []
    for m, psi in enumerate(pacf, start=1):
        phi = [phi[j] - psi * phi[m - 2 - j] for j in range(m - 1)] + [psi]
    return np.array(phi)


def _lyapunov_covariance(n, pacf, tau):
    """Stationary covariance of (x_1, ..., x_n) from the companion state form."""
    p = len(pacf)
    phi = _reference_phi(pacf)
    sigma2_eps = np.prod(1 - np.asarray(pacf) ** 2) / tau

    F = np.zeros((p, p))
    F[0, :] = phi
    F[1:, :-1] = np.eye(p - 1)
    G = np.zeros((p, p))
    G[0, 0] = sigma2_eps

    # P = F P F^T + G, P[0, k] = gamma(k) for k < p
    P = scipy.linalg.solve_discrete_lyapunov(F, G)
    gamma = list(P[0, :])
    for k in range(p, n):
        gamma.append(sum(phi[j] * gamma[k - 1 - j] for j in range(p)))
    gamma = np.array(gamma[:n])

    idx = np.arange(n)
    return gamma[np.abs(idx[:, None] - idx[None, :])]


def _lyapunov_Q(n, pacf, tau):
    return np.linalg.inv(_lyapunov_covariance(n, pacf, tau))


def _logdet_identity(n, pacf, tau):
    pacf = np.asarray(pacf)
    k = np.arange(1, min(len(pacf), n - 1) + 1)
    return n * np.log(tau) - np.sum((n - k) * np.log(1 - pacf[k - 1] ** 2))


# --- Durbin-Levinson ----------------------------------------------------------
def test_pacf_to_ar_coefficients_closed_forms():
    psi1, psi2, psi3 = 0.6, -0.4, 0.3
    levels = ARSubModel._pacf_to_ar_coefficients([psi1, psi2, psi3])

    assert len(levels) == 4
    assert levels[0].size == 0
    np.testing.assert_allclose(levels[1], [psi1])
    np.testing.assert_allclose(levels[2], [psi1 * (1 - psi2), psi2])
    phi21, phi22 = psi1 * (1 - psi2), psi2
    np.testing.assert_allclose(
        levels[3], [phi21 - psi3 * phi22, phi22 - psi3 * phi21, psi3]
    )


@pytest.mark.parametrize("pacf", PACFS_HIGHER_ORDER)
def test_pacf_to_ar_coefficients_is_stationary(pacf):
    phi = ARSubModel._pacf_to_ar_coefficients(pacf)[-1]
    np.testing.assert_allclose(phi, _reference_phi(pacf), rtol=RTOL, atol=ATOL)

    # roots of z^p - phi_1 z^(p-1) - ... - phi_p lie inside the unit circle
    roots = np.roots(np.concatenate(([1.0], -phi)))
    assert np.all(np.abs(roots) < 1.0)


# --- General AR(p) versus the former closed forms -------------------------------
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("phi", PACF_VALUES)
@pytest.mark.parametrize("n", [2, 3, 4, 7, 25])
def test_general_ar1_matches_legacy_formula(tmp_path, n, phi, tau):
    Q = _Q(tmp_path, n, [phi], tau)
    np.testing.assert_allclose(Q, _legacy_ar1_Q(n, phi, tau), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("pacf2", PACF_VALUES)
@pytest.mark.parametrize("pacf1", PACF_VALUES)
@pytest.mark.parametrize("n", [4, 5, 6, 25])
def test_general_ar2_matches_legacy_formula(tmp_path, n, pacf1, pacf2, tau):
    # the comparison is entrywise and therefore covers every boundary entry
    Q = _Q(tmp_path, n, [pacf1, pacf2], tau)
    np.testing.assert_allclose(
        Q, _legacy_ar2_Q(n, pacf1, pacf2, tau), rtol=RTOL, atol=ATOL
    )


# --- Small dimensions ----------------------------------------------------------
def test_general_ar1_singleton_is_tau(tmp_path):
    Q = _dense(_make_ar(tmp_path, 1, [0.5]).construct_Q_prior(pacf1=0.5, tau=2.0))
    np.testing.assert_allclose(Q, [[2.0]], rtol=RTOL)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_general_ar2_accepts_small_n(tmp_path, n):
    pacf, tau = [0.6, -0.4], 1.7
    Q = _dense(_make_ar(tmp_path, n, pacf).construct_Q_prior(**_pacf_kwargs(pacf, tau)))
    np.testing.assert_allclose(Q, _lyapunov_Q(n, pacf, tau), rtol=1e-8, atol=1e-8)


# --- Reductions ---------------------------------------------------------------
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("phi", PACF_VALUES)
def test_ar2_with_zero_pacf2_reduces_to_ar1(tmp_path, phi, tau):
    n = 9
    Q = _Q(tmp_path, n, [phi, 0.0], tau)
    np.testing.assert_allclose(Q, _legacy_ar1_Q(n, phi, tau), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("order", [3, 4, 6])
@pytest.mark.parametrize("n", [2, 5, 12])
def test_trailing_zero_pacfs_reduce_order(tmp_path, n, order):
    pacf, tau = [0.6, -0.4], 3.0
    padded = pacf + [0.0] * (order - 2)

    Q = _Q(tmp_path, n, padded, tau)
    Q_ar2 = _Q(tmp_path, n, pacf, tau)
    np.testing.assert_allclose(Q, Q_ar2, rtol=RTOL, atol=ATOL)
    if n >= 4:
        np.testing.assert_allclose(
            Q, _legacy_ar2_Q(n, *pacf, tau), rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize("n", [1, 3, 10])
def test_all_zero_pacfs_give_scaled_identity(tmp_path, n, order):
    tau = 4.2
    Q = _Q(tmp_path, n, [0.0] * order, tau)
    np.testing.assert_allclose(Q, tau * np.eye(n), rtol=RTOL, atol=ATOL)


# --- Higher orders versus the companion-state Lyapunov covariance -------------------
@pytest.mark.parametrize("tau", [0.5, 8.0])
@pytest.mark.parametrize("pacf", PACFS_HIGHER_ORDER)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 6, 7, 20])
def test_higher_order_matches_lyapunov_reference(tmp_path, n, pacf, tau):
    # covers n <= p, n = p + 1 and n > 2p (boundary blocks overlapping or not)
    Q = _Q(tmp_path, n, pacf, tau)

    Sigma = _lyapunov_covariance(n, pacf, tau)
    np.testing.assert_allclose(Q @ Sigma, np.eye(n), rtol=0, atol=1e-8)
    np.testing.assert_allclose(Q, _lyapunov_Q(n, pacf, tau), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("tau", [0.5, 8.0])
@pytest.mark.parametrize("pacf", [[0.7], [0.6, -0.4]] + PACFS_HIGHER_ORDER)
@pytest.mark.parametrize("n", [1, 4, 7, 30])
def test_matrix_properties(tmp_path, n, pacf, tau):
    p = len(pacf)
    Q = _Q(tmp_path, n, pacf, tau)

    # symmetry and positive definiteness
    np.testing.assert_allclose(Q, Q.T, rtol=0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Q) > 0)

    # half-bandwidth p
    i, j = np.nonzero(Q)
    assert np.max(np.abs(i - j), initial=0) <= p

    # marginal (not innovation) precision
    np.testing.assert_allclose(np.diag(np.linalg.inv(Q)), 1 / tau, rtol=1e-8)

    # exact determinant
    _, logdet = np.linalg.slogdet(Q)
    np.testing.assert_allclose(
        logdet, _logdet_identity(n, pacf, tau), rtol=1e-9, atol=1e-9
    )


def test_innovation_precision_in_the_interior(tmp_path):
    # the interior diagonal is (1 + sum phi^2) / sigma_eps^2
    pacf, tau, n = [0.6, -0.4, 0.3], 2.0, 12
    phi = _reference_phi(pacf)
    sigma2_eps = np.prod(1 - np.array(pacf) ** 2) / tau

    Q = _Q(tmp_path, n, pacf, tau)
    np.testing.assert_allclose(Q[5, 5], (1 + np.sum(phi**2)) / sigma2_eps, rtol=RTOL)
    np.testing.assert_allclose(Q[-1, -1], 1 / sigma2_eps, rtol=RTOL)
    np.testing.assert_allclose(Q[-1, -4], -phi[2] / sigma2_eps, rtol=RTOL)


# --- Sparse output conventions --------------------------------------------------
@pytest.mark.parametrize("order, n", [(1, 1), (1, 6), (2, 6), (3, 2), (4, 9)])
def test_sparsity_pattern_is_parameter_independent(tmp_path, order, n):
    ar = _make_ar(tmp_path, n, [0.0] * order)
    bw = min(order, n - 1)

    Q_a = ar.construct_Q_prior(**_pacf_kwargs([0.3] * order, 1.0))
    # zeros are stored explicitly
    Q_b = ar.construct_Q_prior(**_pacf_kwargs([0.0] * order, 5.0))

    assert Q_a.format == "coo"
    for Q in (Q_a, Q_b):
        row, col = get_host(Q.row), get_host(Q.col)
        assert Q.data.size == n + 2 * sum(n - k for k in range(1, bw + 1))
        np.testing.assert_array_equal(row, get_host(Q_a.row))
        np.testing.assert_array_equal(col, get_host(Q_a.col))
        # row-major sorted without duplicates, as Model's data mapping expects
        flat = row.astype(np.int64) * n + col
        assert np.all(np.diff(flat) > 0)
        assert np.max(np.abs(row - col), initial=0) == bw


# --- Validation ---------------------------------------------------------------
@pytest.mark.parametrize(
    "pacf, tau",
    [
        ([1.0, 0.2], 1.0),
        ([0.2, -1.0], 1.0),
        ([1.5, 0.2], 1.0),
        ([np.nan, 0.2], 1.0),
        ([0.2, np.inf], 1.0),
        ([0.2, 0.1], 0.0),
        ([0.2, 0.1], -3.0),
        ([0.2, 0.1], np.nan),
        ([0.2, 0.1], np.inf),
        ([0.2], 1.0),  # pacf2 missing
    ],
)
def test_construct_Q_prior_rejects_invalid_hyperparameters(tmp_path, pacf, tau):
    ar = _make_ar(tmp_path, 6, [0.0, 0.0])
    with pytest.raises(ValueError):
        ar.construct_Q_prior(**_pacf_kwargs(pacf, tau))


# --- Configuration ------------------------------------------------------------
def test_config_parsing_and_parameter_ordering(tmp_path):
    config = submodels_config.parse_config(_ar_dict(str(tmp_path), [0.6, -0.4, 0.3], 2.0))

    assert isinstance(config, ARSubModelConfig)
    assert config.order == 3
    assert len(config.ph_pacf) == 3
    assert all(ph.type == "beta" and ph.support == (-1.0, 1.0) for ph in config.ph_pacf)
    assert config.ph_tau.type == "gamma"

    theta, theta_keys = config.read_hyperparameters()
    assert theta_keys == ["pacf1", "pacf2", "pacf3", "tau"]
    np.testing.assert_allclose(get_host(theta), [0.6, -0.4, 0.3, 2.0])


@pytest.mark.parametrize("model_type", ["ar1", "ar2"])
def test_removed_submodel_types_are_rejected(tmp_path, model_type):
    with pytest.raises(ValueError, match="Unknown submodel type"):
        submodels_config.parse_config({"type": model_type, "input_dir": str(tmp_path)})


@pytest.mark.parametrize(
    "update",
    [
        {"order": 0, "pacf": [], "ph_pacf": []},
        {"order": 2.0},
        {"order": 3},  # pacf / prior count mismatch
        {"pacf": [0.2]},
        {"ph_pacf": [dict(BETA_PM1)]},
        {"ph_pacf": dict(BETA_PM1)},
        {"pacf": [1.0, 0.1]},
        {"pacf": [0.2, -1.2]},
        {"pacf": [float("nan"), 0.1]},
        {"tau": 0.0},
        {"tau": -1.0},
        {"tau": float("inf")},
        # default beta support is (0, 1): a negative initial pacf is outside
        {"ph_pacf": [{"type": "beta", "alpha": 2.0, "beta": 2.0}] * 2},
    ],
)
def test_config_rejects_invalid_input(tmp_path, update):
    config = _ar_dict(str(tmp_path), [0.2, -0.1])
    config.update(update)
    with pytest.raises(ValueError):
        submodels_config.parse_config(config)


def test_config_requires_all_fields(tmp_path):
    for key in ["order", "pacf", "ph_pacf", "tau", "ph_tau"]:
        config = _ar_dict(str(tmp_path), [0.2, -0.1])
        del config[key]
        with pytest.raises((ValueError, KeyError)):
            submodels_config.parse_config(config)
