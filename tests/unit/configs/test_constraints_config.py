# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dalia.configs.constraints_config import LinearConstraintConfig, parse_config


def test_linear_constraint_coerces_lists_to_arrays():
    cfg = parse_config({"type": "linear", "A": [[1.0, 1.0, 1.0]], "e": [0.0]})
    assert cfg.type == "linear"
    assert isinstance(cfg.A, np.ndarray) and cfg.A.shape == (1, 3)
    assert isinstance(cfg.e, np.ndarray) and cfg.e.shape == (1,)


def test_linear_constraint_accepts_sparse_A():
    cfg = parse_config({"type": "linear", "A": csr_matrix(np.ones((2, 4))), "e": np.zeros(2)})
    assert cfg.A.shape == (2, 4)


def test_linear_constraint_requires_A_and_e():
    with pytest.raises(ValueError):
        parse_config({"type": "linear", "A": [[1.0, 1.0]]})
    with pytest.raises(ValueError):
        parse_config({"type": "linear", "e": [0.0]})


def test_linear_constraint_checks_row_count():
    with pytest.raises(ValueError):
        parse_config({"type": "linear", "A": np.ones((2, 3)), "e": np.zeros(1)})


def test_sum_to_zero_takes_no_data():
    cfg = parse_config({"type": "sum_to_zero"})
    assert cfg.A is None and cfg.e is None
    with pytest.raises(ValueError):
        parse_config({"type": "sum_to_zero", "A": [[1.0]], "e": [0.0]})


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        parse_config({"type": "linear", "A": [[1.0]], "e": [0.0], "b": [0.0]})


def test_parse_config_passes_through_instances():
    cfg = LinearConstraintConfig(type="sum_to_zero")
    assert parse_config(cfg) is cfg
