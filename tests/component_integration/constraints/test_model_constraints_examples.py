from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

from dalia import xp

from dalia.configs import (
    constraints_config,
    dalia_config,
    likelihood_config,
    models_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatialSubModel, SpatioTemporalSubModel

REPO = Path(__file__).resolve().parents[3]
GST_SMALL = REPO / "examples" / "gst_small"
GST_COREG = REPO / "examples" / "gst_coreg2_small" / "inputs_nv2_ns354_nt12_nb2"
GS_COREG = REPO / "examples" / "gs_coreg2_small" / "inputs_nv2_ns1818_nt1_nb2"
GAUSSIAN_PRIOR = {"type": "gaussian", "mean": 0.0, "precision": 0.5}


def build_gst_small(regression_constraints=(), constraints=()):
    """Regression declared first; Model moves the spatio-temporal submodel to the
    front, so the regression block ends up last (columns 460..465)."""
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": str(GST_SMALL / "inputs_spatio_temporal"),
                "spatial_domain_dimension": 2,
                "r_s": 0,
                "r_t": 0,
                "sigma_st": 0,
                "manifold": "sphere",
                "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
                "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
                "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(GST_SMALL / "inputs_regression"),
                "n_fixed_effects": 6,
                "constraints": list(regression_constraints),
            }
        )
    )
    return Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(
            {
                "type": "gaussian",
                "prec_o": 4.0,
                "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0},
            }
        ),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def coregional_variate(i, constraints=(), field_constraints=()):
    variate_dir = GST_COREG / f"model_{i}"
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": str(variate_dir / "inputs_spatio_temporal"),
                "spatial_domain_dimension": 2,
                "r_s": 0.0,
                "r_t": 0.0,
                "sigma_st": 0.0,
                "manifold": "plane",
                "ph_s": dict(GAUSSIAN_PRIOR),
                "ph_t": dict(GAUSSIAN_PRIOR),
                "ph_st": dict(GAUSSIAN_PRIOR),
                "constraints": list(field_constraints),
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(variate_dir / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    return Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(
            {"type": "gaussian", "prec_o": 1.0, "prior_hyperparameters": dict(GAUSSIAN_PRIOR)}
        ),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def spatial_variate(i, constraints=(), field_constraints=()):
    variate_dir = GS_COREG / f"model_{i}"
    spatial = SpatialSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatial",
                "input_dir": str(variate_dir / "inputs_spatial"),
                "spatial_domain_dimension": 2,
                "r_s": 0.0,
                "sigma_e": 0.0,
                "ph_s": dict(GAUSSIAN_PRIOR),
                "ph_e": dict(GAUSSIAN_PRIOR),
                "constraints": list(field_constraints),
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(variate_dir / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    return Model(
        submodels=[spatial, regression],
        likelihood_config=likelihood_config.parse_config(
            {"type": "gaussian", "prec_o": 1.0, "prior_hyperparameters": dict(GAUSSIAN_PRIOR)}
        ),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def coregional_config():
    return models_config.parse_config(
        {
            "type": "coregional",
            "n_models": 2,
            "sigmas": [0.0, 0.0],
            "lambdas": [0.0],
            "ph_sigmas": [dict(GAUSSIAN_PRIOR), dict(GAUSSIAN_PRIOR)],
            "ph_lambdas": [dict(GAUSSIAN_PRIOR)],
        }
    )


def test_submodel_constraint_follows_reordering():
    model = build_gst_small(
        regression_constraints=[{"type": "linear", "C": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]}]
    )

    assert isinstance(model.submodels[0], SpatioTemporalSubModel)
    C = np.asarray(model.C)
    assert C.shape == (1, 466)
    np.testing.assert_array_equal(C[0, :460], np.zeros(460))
    np.testing.assert_array_equal(C[0, 460:], np.arange(1.0, 7.0))


def test_model_constraint_uses_final_latent_order():
    row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    C_model = np.zeros((1, 466))
    C_model[0, 460:] = row

    on_submodel = build_gst_small(regression_constraints=[{"type": "linear", "C": [row]}])
    on_model = build_gst_small(constraints=[{"type": "linear", "C": C_model}])

    np.testing.assert_array_equal(np.asarray(on_model.C), np.asarray(on_submodel.C))


def coregional_model(models, constraints=()):
    return CoregionalModel(
        models=models,
        coregional_model_config=coregional_config(),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def stacked_to_internal(model, x_stacked):
    """x in the internal (permuted) order of a CoregionalModel from x in the
    stacked order (variate 1 block, variate 2 block, ...)."""
    return x_stacked[np.asarray(model.permutation_latent_variables)]


def test_coregional_variate_constraint_follows_spatio_temporal_permutation():
    """The spatio-temporal coregional model interleaves the variates time slice by
    time slice and moves all fixed effects to the end. A sum-to-zero on the field of
    variate 1 must land on the columns of its slices."""
    ns, nt, nb = 354, 12, 1
    constrained = coregional_variate(1, field_constraints=[{"type": "sum_to_zero"}])

    model = coregional_model([constrained, coregional_variate(2)])

    C = np.asarray(model.C)
    assert C.shape == (1, 2 * ns * nt + 2 * nb)
    expected = np.zeros(2 * ns * nt + 2 * nb)
    for t in range(nt):
        expected[t * 2 * ns : t * 2 * ns + ns] = 1.0
    np.testing.assert_array_equal(C[0], expected)
    np.testing.assert_array_equal(np.asarray(model.e), [0.0])


def test_coregional_variate_constraint_follows_spatial_permutation():
    """The spatial coregional model keeps the fields stacked and moves the fixed
    effects of all variates to the end."""
    ns, nb = 1818, 1
    constrained = spatial_variate(2, field_constraints=[{"type": "sum_to_zero"}])

    model = coregional_model([spatial_variate(1), constrained])

    C = np.asarray(model.C)
    assert C.shape == (1, 2 * ns + 2 * nb)
    expected = np.zeros(2 * ns + 2 * nb)
    expected[ns : 2 * ns] = 1.0
    np.testing.assert_array_equal(C[0], expected)


def test_coregional_level_constraint_is_given_in_stacked_order():
    """A constraint declared on the CoregionalModel has its columns in the stacked
    order; the model permutes them so that C x is the same in both orders."""
    n = 2 * 354 * 12 + 2
    rng = np.random.default_rng(3)
    row = rng.normal(size=n)
    e = 2.5

    model = coregional_model(
        [coregional_variate(1), coregional_variate(2)],
        constraints=[{"type": "linear", "C": [row.tolist()], "e": [e]}],
    )

    x_stacked = rng.normal(size=n)
    np.testing.assert_allclose(
        np.asarray(model.C) @ stacked_to_internal(model, x_stacked),
        [row @ x_stacked],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(np.asarray(model.e), [e])


def test_coregional_rows_are_stacked_variates_first_then_model():
    """Rows: the variates' constraints in order, then the coregional-level ones."""
    n = 2 * 354 * 12 + 2
    ns, nt = 354, 12
    row = np.zeros(n)
    row[ns * nt] = 1.0  # intercept of variate 1, stacked order
    row[2 * ns * nt + 1] = -1.0  # intercept of variate 2, stacked order

    model = coregional_model(
        [
            coregional_variate(1, field_constraints=[{"type": "sum_to_zero"}]),
            coregional_variate(2, field_constraints=[{"type": "sum_to_zero"}]),
        ],
        constraints=[{"type": "linear", "C": [row.tolist()]}],
    )

    C = np.asarray(model.C)
    assert C.shape == (3, n)
    # Variate 1 field on even slices, variate 2 field on odd slices
    assert C[0, :ns].sum() == ns and C[0, ns : 2 * ns].sum() == 0
    assert C[1, :ns].sum() == 0 and C[1, ns : 2 * ns].sum() == ns
    # Fixed effects are the last two columns, variate 1 first
    np.testing.assert_array_equal(C[2, -2:], [1.0, -1.0])
    np.testing.assert_array_equal(C[2, :-2], np.zeros(n - 2))


def test_coregional_model_rejects_redundant_constraints():
    n = 2 * 354 * 12 + 2
    row = np.zeros(n)
    row[: 354 * 12] = 1.0  # same as the sum-to-zero of variate 1

    with pytest.raises(ValueError, match="full row rank"):
        coregional_model(
            [coregional_variate(1, field_constraints=[{"type": "sum_to_zero"}]), coregional_variate(2)],
            constraints=[{"type": "linear", "C": [row.tolist()]}],
        )


def test_coregional_model_releases_variate_constraints():
    constrained = coregional_variate(1, constraints=[{"type": "sum_to_zero"}])

    coregional_model([constrained, coregional_variate(2)])

    assert constrained.C is None and constrained.e is None


# --- End to end: DALIA on the spatial coregional example against a dense reference


def make_dalia_config():
    return dalia_config.parse_config(
        {
            "solver": {"type": "dense"},
            "verbosity": 0,
            "minimize": {"max_iter": 10, "gtol": 1e-3, "maxcor": 3},
        }
    )


def observation_precisions(model):
    """Precision of each observation of a Gaussian coregional model."""
    tau = np.zeros(model.n_observations)
    for i in range(model.n_models):
        tau[model.n_observations_idx[i] : model.n_observations_idx[i + 1]] = float(
            model.theta_external[model.hyperparameters_idx[i + 1] - 1]
        )
    return tau


def reference_reduced(model, theta_internal, C, e):
    """f(theta) = -log p(y | theta, Cx = e) - log p(theta) up to a constant, and the
    constrained posterior mean, both computed densely in the stacked order
    (variate 1 block, variate 2 block, ...), independently of the permutation of
    the model: Q and a are mapped back to the stacked order and C is given in it.

    The computation is in the coordinates z of the constraint null space,
    x = x_f + N z with C N = 0 and C x_f = e:
    log p(y | theta) = log p(z) + log p(y | z) - log p(z | y) at the posterior mean
    of z, all three unconstrained Gaussians of dimension n - k.
    """
    model.theta_internal = xp.asarray(theta_internal)
    to_stacked = np.argsort(np.asarray(model.permutation_latent_variables))
    Q = np.asarray(model.construct_Q_prior().toarray())[to_stacked][:, to_stacked]
    a = scipy.sparse.csr_matrix(model.a)[:, to_stacked]
    y = np.asarray(model.y)
    tau = observation_precisions(model)

    N = scipy.linalg.null_space(C)
    x_f = C.T @ np.linalg.solve(C @ C.T, e)
    aTDa = (a.T @ scipy.sparse.diags(tau) @ a).toarray()
    aTDy = a.T @ (tau * y)

    Q_z = N.T @ Q @ N
    m_z = -np.linalg.solve(Q_z, N.T @ (Q @ x_f))
    Q_zp = N.T @ (Q + aTDa) @ N
    m_zp = np.linalg.solve(Q_zp, N.T @ (aTDy - (Q + aTDa) @ x_f))

    x = x_f + N @ m_zp
    r = y - a @ x
    log_prior_z = 0.5 * np.linalg.slogdet(Q_z)[1] - 0.5 * (m_zp - m_z) @ Q_z @ (m_zp - m_z)
    log_likelihood = 0.5 * np.sum(np.log(tau)) - 0.5 * r @ (tau * r)
    log_posterior_z = 0.5 * np.linalg.slogdet(Q_zp)[1]
    log_marginal_likelihood = log_prior_z + log_likelihood - log_posterior_z

    f = -(log_marginal_likelihood + float(model.evaluate_log_prior_hyperparameters()))
    return f, x


def test_dalia_matches_dense_reference_on_constrained_spatial_coregional():
    """Sum-to-zero on both fields and a coregional-level row tying the intercepts,
    with e != 0: f(theta) differences and the constrained mean match a reference
    written in the stacked order."""
    ns = 1818
    row = np.zeros(2 * ns + 2)
    row[ns], row[2 * ns + 1] = 1.0, -1.0  # intercepts, stacked order
    model = coregional_model(
        [
            spatial_variate(1, field_constraints=[{"type": "sum_to_zero"}]),
            spatial_variate(2, field_constraints=[{"type": "sum_to_zero"}]),
        ],
        constraints=[{"type": "linear", "C": [row.tolist()], "e": [0.7]}],
    )
    dalia = DALIA(model=model, config=make_dalia_config())

    # The same constraints, written by hand in the stacked order
    C_stacked = np.zeros((3, 2 * ns + 2))
    C_stacked[0, :ns] = 1.0
    C_stacked[1, ns + 1 : 2 * ns + 1] = 1.0
    C_stacked[2] = row
    e = np.array([0.0, 0.0, 0.7])

    theta_0 = np.asarray(model.theta_internal)
    thetas = [theta_0, theta_0 + 0.3, theta_0 - 0.2]

    f = [float(dalia._evaluate_f(xp.asarray(theta))) for theta in thetas]
    to_stacked = np.argsort(np.asarray(model.permutation_latent_variables))
    x_dalia = np.asarray(model.x)[to_stacked]  # constrained mean at thetas[-1]
    f_ref, x_ref = zip(*[reference_reduced(model, theta, C_stacked, e) for theta in thetas])

    np.testing.assert_allclose(np.diff(f), np.diff(f_ref), rtol=1e-8)
    np.testing.assert_allclose(x_dalia, x_ref[-1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(C_stacked @ x_dalia, e, atol=1e-10)


def test_dalia_matches_dense_reference_on_constrained_spatio_temporal_coregional():
    """Same check on the spatio-temporal coregional model, whose latent variables are
    interleaved time slice by time slice: sum-to-zero on the field of variate 1 and a
    coregional-level row tying the intercepts."""
    ns, nt = 354, 12
    n = 2 * ns * nt + 2
    row = np.zeros(n)
    row[ns * nt], row[n - 1] = 1.0, -1.0  # intercepts, stacked order
    model = coregional_model(
        [
            coregional_variate(1, field_constraints=[{"type": "sum_to_zero"}]),
            coregional_variate(2),
        ],
        constraints=[{"type": "linear", "C": [row.tolist()], "e": [0.7]}],
    )
    dalia = DALIA(model=model, config=make_dalia_config())

    C_stacked = np.zeros((2, n))
    C_stacked[0, : ns * nt] = 1.0
    C_stacked[1] = row
    e = np.array([0.0, 0.7])

    theta_0 = np.asarray(model.theta_internal)
    thetas = [theta_0, theta_0 + 0.3]

    f = [float(dalia._evaluate_f(xp.asarray(theta))) for theta in thetas]
    to_stacked = np.argsort(np.asarray(model.permutation_latent_variables))
    x_dalia = np.asarray(model.x)[to_stacked]  # constrained mean at thetas[-1]
    f_ref, x_ref = zip(*[reference_reduced(model, theta, C_stacked, e) for theta in thetas])

    np.testing.assert_allclose(np.diff(f), np.diff(f_ref), rtol=1e-8)
    np.testing.assert_allclose(x_dalia, x_ref[-1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(C_stacked @ x_dalia, e, atol=1e-10)
