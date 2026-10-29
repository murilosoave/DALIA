# Marginal variances of the linear predictor, diag(A Sigma A^T): the kernel
# against the dense product, and the DALIA method against a dense inverse on a
# Gaussian model with 20 AR1 random effects and an intercept.

import numpy as np
import pytest
import scipy.sparse as sps

from dalia import sp, xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import get_host, row_quadratic_forms
from tests import ATOLS, RTOLS

N = 20

# Internal-scale hyperparameters (phi, tau, prec_o)
THETA = np.array([0.3, 1.0, 4.0])


def random_inputs(m=40, n=15):
    """Sparse A with an empty row and a dense SPD Sigma, both as host arrays."""
    rng = np.random.default_rng(3)
    A = sps.random(m, n, density=0.3, random_state=rng, format="csr")
    A.data[:] = rng.normal(size=A.nnz)
    A = A.tolil()
    A[5, :] = 0.0
    A = sps.csr_matrix(A)
    A.sort_indices()
    B = rng.normal(size=(n, n))
    return A, B @ B.T + n * np.eye(n)


def build_model(path):
    rng = np.random.default_rng(1)
    (path / "inputs_ar1").mkdir()
    (path / "inputs_regression").mkdir()
    sps.save_npz(path / "inputs_ar1" / "a.npz", sps.identity(N, format="csr"))
    sps.save_npz(path / "inputs_regression" / "a.npz", sps.csr_matrix(np.ones((N, 1))))
    np.save(path / "y.npy", 2.0 + rng.normal(size=N))

    ar1 = AR1SubModel(
        config=submodels_config.parse_config(
            {
                "type": "ar1",
                "input_dir": str(path / "inputs_ar1"),
                "phi": 0.5,
                "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
                "tau": 3,
                "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(path / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    return Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(
            {
                "type": "gaussian",
                "prec_o": 4.0,
                "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 0.00005},
            }
        ),
    )


def test_kernel_matches_dense_product():
    A, Sigma = random_inputs()
    upper = sp.sparse.csr_matrix(xp.asarray(np.triu(Sigma)))

    variances = row_quadratic_forms(sp.sparse.csr_matrix(A), upper)

    reference = np.einsum("ij,jk,ik->i", A.toarray(), Sigma, A.toarray())
    np.testing.assert_allclose(get_host(variances), reference, rtol=1e-12, atol=1e-12)


def test_kernel_raises_on_a_missing_covariance_entry():
    A, Sigma = random_inputs()
    # only the diagonal of Sigma is stored, the rows of A need off-diagonal entries
    diagonal = sp.sparse.csr_matrix(xp.asarray(np.diag(np.diag(Sigma))))

    with pytest.raises(ValueError):
        row_quadratic_forms(sp.sparse.csr_matrix(A), diagonal)


@pytest.mark.parametrize("solver", ["dense", "scipy"])
def test_marginal_variances_observations(tmp_path, solver):
    model = build_model(tmp_path)
    config = dalia_config.parse_config({"solver": {"type": solver}, "verbosity": 0})
    dalia = DALIA(model=model, config=config)
    dalia._evaluate_f(xp.asarray(THETA))
    theta_external, x_star = model.theta_external, model.x.copy()

    variances = dalia.get_marginal_variances_observations(theta_external, x_star)

    A = get_host(model.a.toarray())
    Sigma = np.linalg.inv(get_host(model.Q_conditional.toarray()))
    reference = np.einsum("ij,jk,ik->i", A, Sigma, A)
    np.testing.assert_allclose(
        get_host(variances), reference, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )
