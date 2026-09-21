# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Integration of the AR(p) submodel in Model."""

import numpy as np
import pytest
import scipy.sparse

from dalia import xp
from dalia.configs import likelihood_config, submodels_config
from dalia.core.model import Model
from dalia.prior_hyperparameters import (
    BetaPriorHyperparameters,
    GammaPriorHyperparameters,
    GaussianPriorHyperparameters,
)
from dalia.submodels import ARSubModel, RegressionSubModel
from dalia.utils.gpu_utils import get_host

BETA_PM1 = {"type": "beta", "alpha": 2.0, "beta": 2.0, "support": [-1.0, 1.0]}
BETA_01 = {"type": "beta", "alpha": 3.0, "beta": 2.0}
GAMMA = {"type": "gamma", "alpha": 2.0, "beta": 1.0}

N = 12


def _inputs(tmp_path, name, a):
    path = tmp_path / name
    path.mkdir()
    scipy.sparse.save_npz(path / "a.npz", scipy.sparse.csc_matrix(a))
    return str(path)


def _model(tmp_path, ar_dict, ar_class):
    np.save(tmp_path / "y.npy", np.linspace(-1.0, 1.0, N))

    ar_dict["input_dir"] = _inputs(tmp_path, "inputs_ar", scipy.sparse.eye(N))
    ar = ar_class(config=submodels_config.parse_config(ar_dict))

    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": _inputs(tmp_path, "inputs_regression", np.ones((N, 1))),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )

    likelihood = likelihood_config.parse_config(
        {
            "type": "gaussian",
            "prec_o": 20.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 0.01},
        }
    )
    return Model(submodels=[ar, regression], likelihood_config=likelihood)


def _ar3_dict(**update):
    ar_dict = {
        "type": "ar",
        "order": 3,
        "pacf": [0.6, -0.4, 0.3],
        "ph_pacf": [dict(BETA_PM1), dict(BETA_PM1), dict(BETA_PM1)],
        "tau": 2.0,
        "ph_tau": dict(GAMMA),
    }
    ar_dict.update(update)
    return ar_dict


def test_parameter_ordering_and_prior_assignment(tmp_path):
    # pacf2 uses the default beta support (0, 1), the others (-1, 1)
    model = _model(
        tmp_path,
        _ar3_dict(
            pacf=[0.6, 0.4, -0.3],
            ph_pacf=[dict(BETA_PM1), dict(BETA_01), dict(BETA_PM1)],
            ph_tau={"type": "gaussian", "mean": 1.0, "precision": 0.5},
        ),
        ARSubModel,
    )

    assert model.theta_keys == ["pacf1", "pacf2", "pacf3", "tau", "prec_o"]
    assert model.hyperparameters_idx == [0, 4, 4]
    np.testing.assert_allclose(
        get_host(model.theta_external), [0.6, 0.4, -0.3, 2.0, 20.0]
    )

    priors = model.prior_hyperparameters
    assert [type(prior) for prior in priors] == [
        BetaPriorHyperparameters,
        BetaPriorHyperparameters,
        BetaPriorHyperparameters,
        GaussianPriorHyperparameters,
        GammaPriorHyperparameters,
    ]
    assert [(prior.lower, prior.upper) for prior in priors[:3]] == [
        (-1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
    ]
    assert priors[1].alpha == 3.0

    assert np.isfinite(float(model.evaluate_log_prior_hyperparameters()))


def test_transformation_round_trip(tmp_path):
    model = _model(
        tmp_path,
        _ar3_dict(
            pacf=[0.6, 0.4, -0.3],
            ph_pacf=[dict(BETA_PM1), dict(BETA_01), dict(BETA_PM1)],
        ),
        ARSubModel,
    )
    theta_external = get_host(model.theta_external)
    theta_internal = get_host(model.theta_internal)

    # support (-1, 1): z = 12 log((1 + psi) / (1 - psi)), psi = tanh(z / 24)
    for i in (0, 2):
        psi = theta_external[i]
        np.testing.assert_allclose(
            theta_internal[i], 12 * np.log((1 + psi) / (1 - psi)), rtol=1e-12
        )
        np.testing.assert_allclose(np.tanh(theta_internal[i] / 24), psi, rtol=1e-12)
    # default support (0, 1): z = 12 log(psi / (1 - psi))
    np.testing.assert_allclose(
        theta_internal[1], 12 * np.log(0.4 / 0.6), rtol=1e-12
    )
    # gamma prior: log scale
    np.testing.assert_allclose(theta_internal[3], np.log(2.0), rtol=1e-12)

    model.theta_internal = model.theta_internal
    np.testing.assert_allclose(
        get_host(model.theta_external), theta_external, rtol=1e-12
    )

    # jacobians of the (-1, 1) transform are consistent with each other and
    # with a finite difference of the forward map
    prior = model.prior_hyperparameters[0]
    psi, h = 0.6, 1e-6
    forward_jacobian = prior.rescale_hyperparameters_to_internal(
        psi, direction="forward_jacobian"
    )
    backward_jacobian = prior.rescale_hyperparameters_to_internal(
        psi, direction="backward_jacobian"
    )
    finite_difference = (
        prior.rescale_hyperparameters_to_internal(psi + h, direction="forward")
        - prior.rescale_hyperparameters_to_internal(psi - h, direction="forward")
    ) / (2 * h)
    np.testing.assert_allclose(
        float(forward_jacobian), 24 / (1 - psi**2), rtol=1e-12
    )
    np.testing.assert_allclose(
        float(forward_jacobian * backward_jacobian), 1.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(forward_jacobian), float(finite_difference), rtol=1e-6
    )


def test_Q_prior_update_keeps_pattern_through_zero_pacfs(tmp_path):
    model = _model(tmp_path, _ar3_dict(), ARSubModel)
    ar = model.submodels[0]

    def expected(pacf, tau):
        Q = np.zeros((N + 1, N + 1))
        kwargs = {f"pacf{k}": v for k, v in enumerate(pacf, start=1)}
        Q[:N, :N] = get_host(ar.construct_Q_prior(tau=tau, **kwargs).toarray())
        Q[N, N] = 0.001
        return Q

    Q = model.construct_Q_prior()
    nnz = Q.nnz
    np.testing.assert_allclose(
        get_host(Q.toarray()), expected([0.6, -0.4, 0.3], 2.0), rtol=1e-12
    )

    # exact zeros and a sign change must neither alter the pattern nor the mapping
    for pacf, tau in [([0.0, 0.0, 0.0], 1.0), ([-0.5, 0.0, 0.7], 3.0)]:
        theta = get_host(model.theta_external)
        theta[:3] = pacf
        theta[3] = tau
        model.theta_external = xp.asarray(theta)

        Q = model.construct_Q_prior()
        assert Q.nnz == nnz
        np.testing.assert_allclose(
            get_host(Q.toarray()), expected(pacf, tau), rtol=1e-12, atol=1e-14
        )


@pytest.mark.parametrize(
    "ph_pacf",
    [
        {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        {"type": "gamma", "alpha": 2.0, "beta": 1.0},
    ],
)
def test_unsupported_pacf_prior_is_rejected(tmp_path, ph_pacf):
    ar_dict = _ar3_dict(ph_pacf=[dict(BETA_PM1), ph_pacf, dict(BETA_PM1)])
    with pytest.raises(ValueError, match="pacf2"):
        _model(tmp_path, ar_dict, ARSubModel)


def test_unsupported_tau_prior_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ph_tau"):
        _model(tmp_path, _ar3_dict(ph_tau=dict(BETA_01)), ARSubModel)
