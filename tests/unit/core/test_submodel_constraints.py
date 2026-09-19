# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest

from dalia.configs import submodels_config
from dalia.submodels import AR1SubModel
from dalia.utils import get_host
from tests.constraints_utils import N_LATENT_AR1, ar1_dict, make_dataset


@pytest.fixture
def root(tmp_path):
    return make_dataset(tmp_path)


def test_submodel_without_constraints_has_empty_list(root):
    sm = AR1SubModel(config=submodels_config.parse_config(ar1_dict(root)))
    assert sm.constraints == []
    assert sm.null_space_constraint is None
    assert sm.intrinsic is False
    assert sm.null_space() is None
    with pytest.raises(NotImplementedError):
        sm.logdet_Q_prior_generalized(tau=1.0)


def test_submodel_constraints_from_config(root):
    n = N_LATENT_AR1
    cfg = submodels_config.parse_config(
        ar1_dict(
            root,
            constraints=[
                {"type": "sum_to_zero"},
                {"type": "linear", "A": np.eye(n)[:1], "e": [1.0]},
            ],
        )
    )
    sm = AR1SubModel(config=cfg)
    assert len(sm.constraints) == 2
    np.testing.assert_array_equal(get_host(sm.constraints[0].A.toarray()), np.ones((1, n)))
    np.testing.assert_array_equal(get_host(sm.constraints[1].A.toarray()), np.eye(n)[:1])
    np.testing.assert_array_equal(get_host(sm.constraints[1].e), [1.0])
    assert sm.constraints[1].labels == ["ar1 constraint 1 row 0"]


def test_intrinsic_without_null_space_is_rejected(root):
    class Broken(AR1SubModel):
        intrinsic = True

    with pytest.raises(NotImplementedError, match="null_space"):
        Broken(config=submodels_config.parse_config(ar1_dict(root)))


def test_wrong_width_constraint_is_rejected_at_construction(root):
    cfg = submodels_config.parse_config(
        ar1_dict(root, constraints=[{"type": "linear", "A": [[1.0]], "e": [0.0]}])
    )
    with pytest.raises(ValueError, match="columns"):
        AR1SubModel(config=cfg)


def test_unknown_constraint_type_is_rejected(root):
    with pytest.raises(ValueError):
        submodels_config.parse_config(ar1_dict(root, constraints=[{"type": "quadratic"}]))
