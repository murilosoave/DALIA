# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest

from dalia import xp
from dalia.configs import likelihood_config, submodels_config
from dalia.core.model import Model
from dalia.submodels import AR1SubModel, RegressionSubModel, RW1SubModel
from dalia.utils import get_host
from tests.constraints_utils import (
    N_LATENT_AR1,
    ar1_dict,
    likelihood_dict,
    make_dataset,
    regression_dict,
    rw1_dict,
)

N_TOTAL = N_LATENT_AR1 + 1


def _model(root, ar1_constraints=None, model_constraints=None, ar1_cls=AR1SubModel, tau=1.0):
    if ar1_cls is RW1SubModel:
        cfg = submodels_config.parse_config(rw1_dict(root, tau=tau, constraints=ar1_constraints or []))
    else:
        cfg = submodels_config.parse_config(ar1_dict(root, tau=tau, constraints=ar1_constraints or []))
    ar1 = ar1_cls(config=cfg)
    reg = RegressionSubModel(config=submodels_config.parse_config(regression_dict(root)))
    return Model(
        submodels=[ar1, reg],
        likelihood_config=likelihood_config.parse_config(likelihood_dict("gaussian")),
        constraints=model_constraints,
    )


@pytest.fixture
def root(tmp_path):
    return make_dataset(tmp_path)


def test_model_without_constraints(root):
    model = _model(root)
    assert model.constraints is None
    assert model.constraint_prior_rows is None
    assert model.intrinsic_submodels == []
    Q = model.construct_Q_prior()
    assert model.Q_prior_solver is Q
    assert model.logdet_Q_prior_generalized() == 0.0


def test_model_stacks_submodel_and_model_constraints(root):
    model = _model(
        root,
        ar1_constraints=[{"type": "sum_to_zero"}],
        model_constraints=[{"type": "linear", "A": np.eye(N_TOTAL)[-1:], "e": [0.5]}],
    )
    c = model.constraints
    assert c.k == 2 and c.n == N_TOTAL
    A = get_host(c.A.toarray())
    np.testing.assert_array_equal(A[0], np.r_[np.ones(N_LATENT_AR1), 0.0])
    np.testing.assert_array_equal(A[1], np.r_[np.zeros(N_LATENT_AR1), 1.0])
    np.testing.assert_array_equal(get_host(c.e), [0.0, 0.5])
    np.testing.assert_array_equal(model.constraint_prior_rows, [True, True])
    assert c.labels == ["ar1 constraint 0 (sum_to_zero)", "model constraint 0 row 0"]


def test_model_level_constraint_must_span_all_latent_parameters(root):
    with pytest.raises(ValueError, match="columns"):
        _model(root, model_constraints=[{"type": "linear", "A": np.ones((1, N_LATENT_AR1)), "e": [0.0]}])


def test_dependent_rows_across_levels_are_rejected(root):
    A = np.r_[np.ones(N_LATENT_AR1), 0.0][None, :]
    with pytest.raises(ValueError, match="model constraint 0 row 0"):
        _model(root, ar1_constraints=[{"type": "sum_to_zero"}], model_constraints=[{"type": "linear", "A": A, "e": [0.0]}])


def test_proper_model_cached_construction_matches_fresh(root):
    model = _model(root, ar1_constraints=[{"type": "sum_to_zero"}])
    model.construct_Q_prior()
    theta = model.theta_external
    theta[1] = 2.5
    model.theta_external = theta
    Q_cached = get_host(model.construct_Q_prior().toarray())
    fresh = _model(root, tau=2.5)
    np.testing.assert_allclose(Q_cached, get_host(fresh.construct_Q_prior().toarray()), rtol=1e-14)


def test_str_mentions_constraints(root):
    text = str(_model(root, ar1_constraints=[{"type": "sum_to_zero"}]))
    assert "Number of Constraints" in text


def test_duplicate_sum_to_zero_on_intrinsic_is_rejected(root):
    with pytest.raises(NotImplementedError, match="null-space"):
        _model(root, ar1_constraints=[{"type": "sum_to_zero"}], ar1_cls=RW1SubModel)


def test_model_constraint_touching_intrinsic_block_is_rejected(root):
    A = np.zeros((1, N_TOTAL))
    A[0, 0] = 1.0
    with pytest.raises(NotImplementedError, match="model constraint 0"):
        _model(root, model_constraints=[{"type": "linear", "A": A, "e": [0.0]}], ar1_cls=RW1SubModel)


def test_intrinsic_block_uses_identity_in_solver_copy(root):
    model = _model(root, ar1_cls=RW1SubModel)
    assert model.intrinsic_submodels == [0]
    assert model.constraints.k == 1
    np.testing.assert_array_equal(model.constraint_prior_rows, [False])

    Q = get_host(model.construct_Q_prior().toarray())
    Qs = get_host(model.Q_prior_solver.toarray())
    n = N_LATENT_AR1
    # true prior keeps the singular RW1 block; solver copy has the identity there
    np.testing.assert_allclose(Q[:n, :n] @ np.ones(n), 0.0, atol=1e-12)
    np.testing.assert_array_equal(Qs[:n, :n], np.eye(n))
    np.testing.assert_array_equal(Qs[n:, n:], Q[n:, n:])
    assert model.logdet_Q_prior_generalized() == pytest.approx(np.log(n))  # tau = 1

    # cached construction at a new tau: true prior scales, solver copy stays identity
    theta = model.theta_external
    theta[0] = 2.5
    model.theta_external = theta
    Q2 = get_host(model.construct_Q_prior().toarray())
    np.testing.assert_allclose(Q2[:n, :n], 2.5 * Q[:n, :n])
    np.testing.assert_array_equal(get_host(model.Q_prior_solver.toarray()), Qs)
    assert model.logdet_Q_prior_generalized() == pytest.approx((n - 1) * np.log(2.5) + np.log(n))
