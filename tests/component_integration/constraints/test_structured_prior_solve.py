# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Review point 1: the prior inverse action must be complete on the structured solver,
whose 'bt' factorization covers only the spatio-temporal blocks while the regression
prior lives in the arrow tip. Compared against dense and scipy on examples/gst_small."""

from pathlib import Path

import numpy as np
import pytest

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import get_host

pytest.importorskip("serinv")

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "gst_small"
if not (EXAMPLE / "inputs_spatio_temporal" / "a.npz").exists():
    pytest.skip("examples/gst_small inputs not available", allow_module_level=True)

NS, NT, NB = 92, 5, 6
N = NS * NT + NB
SOLVERS = ["dense", "scipy", "serinv"]


def _host(a):
    return np.asarray(get_host(a))


def temporal_row():
    """Sum-to-zero over the first time slice of the spatio-temporal field."""
    A = np.zeros((1, N))
    A[0, :NS] = 1.0
    return A, np.array([0.0])


def regression_row():
    A = np.zeros((1, N))
    A[0, NS * NT] = 1.0
    return A, np.array([0.3])


def mixed_rows():
    A = np.zeros((3, N))
    A[0, :NS] = 1.0
    A[1, NS * NT] = 1.0
    A[2, NS * (NT - 1) : NS * NT] = 1.0
    A[2, NS * NT + 1] = -1.0
    return A, np.array([0.0, 0.3, 0.1])


CASES = {"temporal": temporal_row, "regression": regression_row, "mixed": mixed_rows}


def build(solver, A=None, e=None):
    st = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": f"{EXAMPLE}/inputs_spatio_temporal",
                "spatial_domain_dimension": 2,
                "r_s": 0.0,
                "r_t": 0.0,
                "sigma_st": 0.0,
                "manifold": "sphere",
                "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
                "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
                "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
            }
        )
    )
    reg = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": f"{EXAMPLE}/inputs_regression",
                "n_fixed_effects": NB,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    model = Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config(
            {"type": "gaussian", "prec_o": 4.0, "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0}}
        ),
        constraints=None if A is None else [{"type": "linear", "A": A, "e": e}],
    )
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(
            {"solver": {"type": solver}, "minimize": {"max_iter": 2}, "simulation_dir": ".", "verbosity": 0}
        ),
    )
    return model, dalia


@pytest.mark.parametrize("case", list(CASES))
def test_prior_solve_and_logdet_agree_across_solvers(case):
    A, e = CASES[case]()
    results = {}
    for solver in SOLVERS:
        model, dalia = build(solver, A, e)
        model.construct_Q_prior()
        dalia.solver.factorize(model.Q_prior_solver, sparsity="bt")
        logdet = float(dalia.solver.logdet(sparsity="bt"))
        _, V, L = dalia._solve_constraints(sparsity="bt")
        results[solver] = (logdet, _host(V).copy(), _host(L).copy())

    Q_p = _host(model.construct_Q_prior().toarray())
    V_ref = np.linalg.solve(Q_p, A.T)
    for solver in SOLVERS:
        logdet, V, L = results[solver]
        np.testing.assert_allclose(logdet, np.linalg.slogdet(Q_p)[1], rtol=1e-10)
        np.testing.assert_allclose(V, V_ref, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(L @ L.T, A @ V_ref, rtol=1e-8)


def test_bt_solve_with_bt_sized_rhs_still_works():
    """A right-hand side without the tip rows (pure BT use) must be accepted."""
    model, dalia = build("serinv")
    Q = model.construct_Q_prior()
    dalia.solver.factorize(Q, sparsity="bt")
    rhs = xp.ones(NS * NT, dtype=xp.float64)
    sol = _host(dalia.solver.solve(rhs=rhs, sparsity="bt"))
    Q_bt = _host(Q.toarray())[: NS * NT, : NS * NT]
    np.testing.assert_allclose(Q_bt @ sol, np.ones(NS * NT), rtol=1e-8)


@pytest.mark.parametrize("case", list(CASES))
def test_constrained_objective_agrees_across_solvers(case):
    A, e = CASES[case]()
    values, means = {}, {}
    for solver in SOLVERS:
        model, dalia = build(solver, A, e)
        # switch prior / conditional factorizations twice to exercise the mode changes
        for _ in range(2):
            values[solver] = float(dalia._evaluate_f(model.theta_internal))
        means[solver] = _host(model.x).copy()
        np.testing.assert_allclose(A @ means[solver], e, atol=1e-8)
    for solver in SOLVERS[1:]:
        np.testing.assert_allclose(values[solver], values["dense"], rtol=1e-9)
        np.testing.assert_allclose(means[solver], means["dense"], rtol=1e-6, atol=1e-8)


def test_unconstrained_objective_unchanged_across_solvers():
    values = {}
    for solver in SOLVERS:
        model, dalia = build(solver)
        values[solver] = float(dalia._evaluate_f(model.theta_internal))
    for solver in SOLVERS[1:]:
        np.testing.assert_allclose(values[solver], values["dense"], rtol=1e-9)
