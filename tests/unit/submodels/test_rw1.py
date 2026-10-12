import numpy as np
import pytest
import scipy.sparse as sps
from pydantic import ValidationError

from dalia.configs import submodels_config
from dalia.submodels import AR1SubModel, AR2SubModel, RegressionSubModel, RW1SubModel

N = 12


def build(tmp_path, n=N, **extra):
    sps.save_npz(tmp_path / "a.npz", sps.identity(n, format="csr"))
    config = {
        "type": "rw1",
        "input_dir": str(tmp_path),
        "tau": 2.0,
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        **extra,
    }
    return RW1SubModel(config=submodels_config.parse_config(config))


def test_Q_prior_is_tau_R_plus_diagonal(tmp_path):
    submodel = build(tmp_path, diagonal=1e-3)
    D = np.diff(np.eye(N), axis=0)

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * D.T @ D + 1e-3 * np.eye(N), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("diagonal", [0.0, -1e-6])
def test_non_positive_diagonal_is_rejected(tmp_path, diagonal):
    """Without the diagonal the precision is singular and cannot be factorized."""
    with pytest.raises(ValidationError):
        build(tmp_path, diagonal=diagonal)


def test_given_constraints_come_after_the_sum_to_zero(tmp_path):
    row = np.zeros((1, N))
    row[0, 0] = 1.0
    constraints = [{"type": "linear", "C": row.tolist(), "e": [0.5]}]

    C, e = build(tmp_path, constraints=constraints).construct_constraints()

    np.testing.assert_array_equal(np.asarray(C), np.vstack([np.ones((1, N)), row]))
    np.testing.assert_array_equal(np.asarray(e), np.array([0.0, 0.5]))


def test_only_intrinsic_submodels_are_wrapped():
    """The diagonal is added by SubModel, through a wrapper around the
    construct_Q_prior of the submodels with intrinsic = True."""
    assert RW1SubModel.intrinsic
    assert RW1SubModel.construct_Q_prior.__name__ == "wrapper"
    for submodel_class in (AR1SubModel, AR2SubModel, RegressionSubModel):
        assert not submodel_class.intrinsic
        assert submodel_class.construct_Q_prior.__name__ == "construct_Q_prior"


def test_submodels_have_no_identifiability_constraints_by_default(tmp_path):
    for submodel_class in (AR1SubModel, AR2SubModel, RegressionSubModel):
        assert submodel_class._construct_identifiability_constraints(None) == []

    assert len(build(tmp_path)._construct_identifiability_constraints()) == 1
