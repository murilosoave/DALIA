# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import get_host
from tests.constraints_utils import (
    N_LATENT_AR1,
    ar1_dict,
    likelihood_dict,
    make_dataset,
    regression_dict,
)

N_TOTAL = N_LATENT_AR1 + 1
DALIA_DICT = {
    "solver": {"type": "dense"},
    "minimize": {"max_iter": 5},
    "simulation_dir": ".",
    "verbosity": 0,
}


def _host(a):
    return np.asarray(get_host(a))


def build(root, likelihood, ar1_constraints=None, model_constraints=None, ar1_cls=AR1SubModel, solver="dense"):
    ar1 = ar1_cls(config=submodels_config.parse_config(ar1_dict(root, constraints=ar1_constraints or [])))
    reg = RegressionSubModel(config=submodels_config.parse_config(regression_dict(root)))
    model = Model(
        submodels=[ar1, reg],
        likelihood_config=likelihood_config.parse_config(likelihood_dict(likelihood, root)),
        constraints=model_constraints,
    )
    cfg = dict(DALIA_DICT)
    cfg["solver"] = {"type": solver}
    dalia = DALIA(model=model, config=dalia_config.parse_config(cfg))
    return model, dalia


def log_normal(v, mean, cov):
    return multivariate_normal(mean=mean, cov=cov).logpdf(v)


def dense_gaussian_pieces(model):
    """Q_prior, Q_conditional and the unconstrained posterior mean for the Gaussian model, dense."""
    prec_o = float(model.theta_external[-1])
    a = _host(model.a.toarray())
    Q_p = _host(model.construct_Q_prior().toarray())
    Q_c = Q_p + prec_o * a.T @ a
    x_star = np.linalg.solve(Q_c, prec_o * a.T @ _host(model.y))
    return Q_p, Q_c, x_star


# --- Gaussian likelihood -------------------------------------------------------


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("e_value", [0.0, 0.4], ids=["e=0", "e!=0"])
def test_gaussian_objective_obeys_bayes_identity(tmp_path, e_value, solver):
    """f_c = f_u - log N(e; A x*, A Q_c^{-1} A^T) + log N(e; 0, A Q_p^{-1} A^T)."""
    root = make_dataset(tmp_path, likelihood="gaussian")
    A = np.r_[np.ones(N_LATENT_AR1), 0.0][None, :]
    e = np.array([e_value])
    model_u, dalia_u = build(root, "gaussian", solver=solver)
    model_c, dalia_c = build(root, "gaussian", model_constraints=[{"type": "linear", "A": A, "e": e}], solver=solver)

    theta = model_u.theta_internal
    f_u = float(dalia_u._evaluate_f(theta))
    f_c = float(dalia_c._evaluate_f(theta))

    Q_p, Q_c, x_star = dense_gaussian_pieces(model_u)
    Sigma_p, Sigma_c = np.linalg.inv(Q_p), np.linalg.inv(Q_c)
    expected = f_u - log_normal(e, A @ x_star, A @ Sigma_c @ A.T) + log_normal(e, np.zeros(1), A @ Sigma_p @ A.T)
    np.testing.assert_allclose(f_c, expected, rtol=1e-8)

    # the stored latent vector is the constrained posterior mean
    x_c = x_star - Sigma_c @ A.T @ np.linalg.solve(A @ Sigma_c @ A.T, A @ x_star - e)
    np.testing.assert_allclose(_host(model_c.x), x_c, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(A @ _host(model_c.x), e, atol=1e-10)


def test_two_constraints_across_blocks(tmp_path):
    """k = 2: sum-to-zero on the AR1 block and a pinned intercept."""
    root = make_dataset(tmp_path, likelihood="gaussian")
    A = np.zeros((2, N_TOTAL))
    A[0, :N_LATENT_AR1] = 1.0
    A[1, -1] = 1.0
    e = np.array([0.0, 0.5])
    model_u, dalia_u = build(root, "gaussian")
    model_c, dalia_c = build(root, "gaussian", ar1_constraints=[{"type": "sum_to_zero"}],
                             model_constraints=[{"type": "linear", "A": A[1:], "e": e[1:]}])
    assert model_c.constraints.k == 2
    theta = model_u.theta_internal
    f_u, f_c = float(dalia_u._evaluate_f(theta)), float(dalia_c._evaluate_f(theta))
    Q_p, Q_c, x_star = dense_gaussian_pieces(model_u)
    Sigma_p, Sigma_c = np.linalg.inv(Q_p), np.linalg.inv(Q_c)
    expected = f_u - log_normal(e, A @ x_star, A @ Sigma_c @ A.T) + log_normal(e, np.zeros(2), A @ Sigma_p @ A.T)
    np.testing.assert_allclose(f_c, expected, rtol=1e-8)
    np.testing.assert_allclose(A @ _host(model_c.x), e, atol=1e-10)


def test_submodel_sum_to_zero_equals_model_level_constraint(tmp_path):
    root = make_dataset(tmp_path, likelihood="gaussian")
    A = np.r_[np.ones(N_LATENT_AR1), 0.0][None, :]
    model_s, dalia_s = build(root, "gaussian", ar1_constraints=[{"type": "sum_to_zero"}])
    model_m, dalia_m = build(root, "gaussian", model_constraints=[{"type": "linear", "A": A, "e": [0.0]}])
    theta = model_s.theta_internal
    np.testing.assert_allclose(float(dalia_s._evaluate_f(theta)), float(dalia_m._evaluate_f(theta)), rtol=1e-12)


def test_unconstrained_model_allocates_no_workspace(tmp_path):
    root = make_dataset(tmp_path, likelihood="gaussian")
    _, dalia = build(root, "gaussian")
    assert dalia._constraint_rhs is None


def test_workspace_is_allocated_once(tmp_path):
    root = make_dataset(tmp_path, likelihood="gaussian")
    model, dalia = build(root, "gaussian", ar1_constraints=[{"type": "sum_to_zero"}])
    assert dalia._constraint_rhs.shape == (N_TOTAL, 2)
    assert dalia._constraint_rhs.flags["F_CONTIGUOUS"]
    buffer = dalia._constraint_rhs
    dalia._evaluate_f(model.theta_internal)
    assert dalia._constraint_rhs is buffer


# --- Non-Gaussian likelihood -------------------------------------------------------


def dense_constrained_laplace_f(model, A, e):
    """Independent dense reference of the constrained Laplace objective at the current theta.

    Constrained Newton (each unconstrained update projected onto {Ax = e}), then
    f = -(log p(theta) + log lik(x) + log p(x | Ax=e) - log p_G(x | y, Ax=e)) with the
    Gaussian approximation p_G at the constrained mode.
    """
    a = _host(model.a.toarray())
    Q_p = _host(model.construct_Q_prior().toarray())
    y = _host(model.y)
    x = A.T @ np.linalg.solve(A @ A.T, e)
    for _ in range(200):
        eta = a @ x
        grad = _host(model.likelihood.evaluate_gradient_likelihood(eta=xp.asarray(eta), y=model.y))
        H = -_host(model.likelihood.evaluate_hessian_likelihood(eta=xp.asarray(eta)).toarray())
        Q_c = Q_p + a.T @ H @ a
        b = -Q_p @ x + a.T @ grad
        x_new = x + np.linalg.solve(Q_c, b)
        Sigma_c = np.linalg.inv(Q_c)
        x_new = x_new - Sigma_c @ A.T @ np.linalg.solve(A @ Sigma_c @ A.T, A @ x_new - e)
        converged = np.linalg.norm(x_new - x) < 1e-12
        x = x_new
        if converged:
            break
    eta = a @ x
    H = -_host(model.likelihood.evaluate_hessian_likelihood(eta=xp.asarray(eta)).toarray())
    Q_c = Q_p + a.T @ H @ a
    Sigma_p, Sigma_c = np.linalg.inv(Q_p), np.linalg.inv(Q_c)
    W_p, W_c = A @ Sigma_p @ A.T, A @ Sigma_c @ A.T

    log_prior_theta = float(model.evaluate_log_prior_hyperparameters())
    lik = float(model.evaluate_likelihood(eta=xp.asarray(eta)))
    prior = (0.5 * np.linalg.slogdet(Q_p)[1] - 0.5 * x @ Q_p @ x
             + 0.5 * np.linalg.slogdet(W_p)[1] + 0.5 * e @ np.linalg.solve(W_p, e))
    cond = 0.5 * np.linalg.slogdet(Q_c)[1] + 0.5 * np.linalg.slogdet(W_c)[1]
    return -(log_prior_theta + lik + prior - cond), x


@pytest.mark.parametrize("e_value", [0.0, 0.4], ids=["e=0", "e!=0"])
def test_poisson_objective_matches_dense_constrained_newton(tmp_path, e_value):
    root = make_dataset(tmp_path, likelihood="poisson")
    A = np.r_[np.ones(N_LATENT_AR1), 0.0][None, :]
    e = np.array([e_value])
    model, dalia = build(root, "poisson", model_constraints=[{"type": "linear", "A": A, "e": e}])
    dalia.eps_inner_iteration = 1e-10

    f_dalia = float(dalia._evaluate_f(model.theta_internal))
    f_ref, x_ref = dense_constrained_laplace_f(model, A, e)

    np.testing.assert_allclose(A @ _host(model.x), e, atol=1e-8)
    np.testing.assert_allclose(_host(model.x), x_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f_dalia, f_ref, rtol=1e-8)


def test_poisson_infeasible_warm_start_with_default_tolerance(tmp_path):
    """Review point 7: a tiny nonzero e with an infeasible start and the default
    eps_inner_iteration must still return a feasible mode."""
    root = make_dataset(tmp_path, likelihood="poisson")
    A = np.zeros((1, N_TOTAL))
    A[0, 0] = 1.0
    e = np.array([1e-4])
    model, dalia = build(root, "poisson", model_constraints=[{"type": "linear", "A": A, "e": e}])
    assert dalia.eps_inner_iteration == 1e-3
    model.x[:] = 0.0  # infeasible: A x = 0 != e
    dalia._evaluate_f(model.theta_internal)
    np.testing.assert_allclose(A @ _host(model.x), e, atol=1e-12)
    # and from a feasible-but-far start
    model.x[:] = xp.asarray(np.r_[1e-4, 0.7 * np.ones(N_TOTAL - 1)])
    dalia._evaluate_f(model.theta_internal)
    np.testing.assert_allclose(A @ _host(model.x), e, atol=1e-12)


def test_poisson_unconstrained_path_unchanged(tmp_path):
    root = make_dataset(tmp_path, likelihood="poisson")
    model, dalia = build(root, "poisson")
    model.construct_Q_prior()
    Q_c, x, eta, constraint_factor = dalia._inner_iteration()
    assert constraint_factor is None
    assert np.isfinite(float(dalia._evaluate_f(model.theta_internal)))


# --- Marginal variances --------------------------------------------------------


def test_gaussian_marginal_variances_are_constrained(tmp_path):
    """diag(Sigma_c) = diag(Q_c^{-1}) - diag(V W^{-1} V^T), for latent parameters and observations."""
    root = make_dataset(tmp_path, likelihood="gaussian")
    A = np.zeros((2, N_TOTAL))
    A[0, :N_LATENT_AR1] = 1.0
    A[1, -1] = 1.0
    e = np.array([0.0, 0.5])
    model, dalia = build(root, "gaussian", model_constraints=[{"type": "linear", "A": A, "e": e}])
    dalia._evaluate_f(model.theta_internal)

    variances = _host(dalia.get_marginal_variances_latent_parameters(model.theta_external, model.x.copy()))
    _, Q_c, _ = dense_gaussian_pieces(model)
    Sigma_c = np.linalg.inv(Q_c)
    Sigma_constrained = Sigma_c - Sigma_c @ A.T @ np.linalg.solve(A @ Sigma_c @ A.T, A @ Sigma_c)
    np.testing.assert_allclose(variances, np.diag(Sigma_constrained), rtol=1e-8, atol=1e-12)

    a = _host(model.a.toarray())
    obs = _host(dalia.get_marginal_variances_observations(model.theta_external, model.x.copy()))
    np.testing.assert_allclose(obs, np.diag(a @ Sigma_constrained @ a.T), rtol=1e-8, atol=1e-12)
