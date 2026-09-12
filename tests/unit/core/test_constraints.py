# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
from dalia import sp, xp
from dalia.configs.constraints_config import parse_config
from dalia.core.constraints import LinearConstraint
from dalia.utils import get_host
from tests import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


def _spd(n: int) -> np.ndarray:
    M = rng.random((n, n))
    return M @ M.T + n * np.eye(n)


def _host(a):
    return np.asarray(get_host(a))


def test_sum_to_zero():
    c = LinearConstraint.sum_to_zero(4)
    assert c.k == 1 and c.n == 4
    assert sp.sparse.issparse(c.A)
    np.testing.assert_array_equal(_host(c.A.toarray()), np.ones((1, 4)))
    np.testing.assert_array_equal(_host(c.e), np.zeros(1))


def test_from_config_linear_and_sum_to_zero():
    c = LinearConstraint.from_config(
        parse_config({"type": "linear", "A": [[1.0, 0.0, -1.0]], "e": [2.0]}), n=3
    )
    np.testing.assert_array_equal(_host(c.A.toarray()), [[1.0, 0.0, -1.0]])
    np.testing.assert_array_equal(_host(c.e), [2.0])
    c2 = LinearConstraint.from_config(parse_config({"type": "sum_to_zero"}), n=5)
    assert c2.A.shape == (1, 5)


def test_from_config_rejects_wrong_width():
    cfg = parse_config({"type": "linear", "A": [[1.0, 1.0]], "e": [0.0]})
    with pytest.raises(ValueError, match="columns"):
        LinearConstraint.from_config(cfg, n=3)


def test_from_null_space_gives_one_row_per_direction():
    N = np.c_[np.ones(4), np.arange(4.0)]
    c = LinearConstraint.from_null_space(N)
    assert c.k == 2
    np.testing.assert_array_equal(_host(c.A.toarray()), N.T)
    np.testing.assert_array_equal(_host(c.e), np.zeros(2))


def test_embed_places_block_at_offset():
    c = LinearConstraint(np.array([[1.0, 2.0]]), np.array([3.0]))
    emb = c.embed(offset=1, n_total=5)
    assert emb.k == 1 and emb.n == 5
    np.testing.assert_array_equal(_host(emb.A.toarray()), [[0.0, 1.0, 2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(_host(emb.e), [3.0])


def test_stack_concatenates_rows_and_labels():
    a = LinearConstraint(np.array([[1.0, 2.0]]), np.array([3.0]), labels=["a"]).embed(1, 5)
    b = LinearConstraint.sum_to_zero(5, label="b")
    st = LinearConstraint.stack([a, b])
    assert st.A.shape == (2, 5)
    np.testing.assert_array_equal(_host(st.A.toarray())[1], np.ones(5))
    np.testing.assert_array_equal(_host(st.e), [3.0, 0.0])
    assert st.labels == ["a", "b"]


def test_subset_and_support_mask():
    c = LinearConstraint(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(c.support_mask(0, 1), [True, False])
    np.testing.assert_array_equal(c.support_mask(1, 3), [False, True])
    s = c.subset(np.array([False, True]))
    assert s.k == 1
    np.testing.assert_array_equal(_host(s.A.toarray()), [[0.0, 0.0, 2.0]])
    np.testing.assert_array_equal(_host(s.e), [2.0])
    empty = c.subset(np.array([False, False]))
    assert empty.k == 0 and empty.n == 3


def test_validate_accepts_independent_rows_regardless_of_scaling():
    A = np.array([[1.0, 1.0, 0.0], [1e-6, 0.0, 1e-6]])
    LinearConstraint(A, np.zeros(2)).validate()


def test_validate_rejects_duplicate_rows_with_labels():
    A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    c = LinearConstraint(A, np.zeros(2), labels=["first", "second"])
    with pytest.raises(ValueError, match="second"):
        c.validate()


def test_validate_rejects_zero_row():
    c = LinearConstraint(np.array([[0.0, 0.0]]), np.zeros(1), labels=["z"])
    with pytest.raises(ValueError, match="all-zero"):
        c.validate()


def test_non_finite_entries_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        LinearConstraint(np.array([[1.0, np.nan]]), np.zeros(1))


def test_project_and_feasible_point():
    c = LinearConstraint(rng.random((2, 6)), rng.random(2))
    x = xp.asarray(rng.random(6))
    np.testing.assert_allclose(_host(c.residual(c.project(x))), 0.0, atol=1e-12)
    np.testing.assert_allclose(_host(c.residual(c.feasible_point())), 0.0, atol=1e-12)
    assert np.all(_host(LinearConstraint.sum_to_zero(6).feasible_point()) == 0.0)


def test_fill_rhs_full_and_subset():
    A = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    c = LinearConstraint(A, np.zeros(2))
    buf = xp.empty((3, 2), dtype=xp.float64, order="F")
    c.fill_rhs(buf)
    np.testing.assert_array_equal(_host(buf), A.T)
    buf1 = xp.empty((3, 1), dtype=xp.float64, order="F")
    c.fill_rhs(buf1, rows=np.array([False, True]))
    np.testing.assert_array_equal(_host(buf1), A[1:].T)


def test_correct_matches_closed_form_and_is_feasible():
    n, k = 6, 2
    Q, A, e, x = _spd(n), rng.random((k, n)), rng.random(k), rng.random(n)
    c = LinearConstraint(A, e)
    Sigma = np.linalg.inv(Q)
    V, W = Sigma @ A.T, A @ Sigma @ A.T
    L = c.factor_W(xp.asarray(W))
    x_c = _host(c.correct(xp.asarray(x), xp.asarray(V), L))
    np.testing.assert_allclose(A @ x_c - e, 0.0, atol=1e-12)
    expected = x - Sigma @ A.T @ np.linalg.solve(W, A @ x - e)
    np.testing.assert_allclose(x_c, expected, rtol=1e-12)


def test_factor_W_rejects_singular_W():
    c = LinearConstraint(np.eye(2), np.zeros(2))
    with pytest.raises(ValueError, match="positive definite"):
        c.factor_W(xp.asarray(np.array([[1.0, 1.0], [1.0, 1.0]])))


def test_variance_correction_matches_dense_in_chunks():
    n, k = 7, 2
    Q, A = _spd(n), rng.random((k, n))
    A[1] *= 1e4  # badly scaled row
    c = LinearConstraint(A, np.zeros(k))
    Sigma = np.linalg.inv(Q)
    V, W = Sigma @ A.T, A @ Sigma @ A.T
    L = c.factor_W(xp.asarray(W))
    Sigma_c = Sigma - V @ np.linalg.solve(W, V.T)
    got = np.diag(Sigma) - _host(c.variance_correction(xp.asarray(V), L, chunk_rows=3))
    np.testing.assert_allclose(got, np.diag(Sigma_c), rtol=1e-12)


def test_log_correction_matches_eigen_reference():
    """Bayes-rule evaluation equals the density of the rank-(n-k) constrained Gaussian
    computed from its non-zero eigenvalues (port of the reference script)."""
    n, k = 5, 1
    Q, mu = _spd(n), rng.random(n)
    A, e, x_unc = rng.random((k, n)), rng.random(k), rng.random(n)
    c = LinearConstraint(A, e)
    Sigma = np.linalg.inv(Q)
    V, W = Sigma @ A.T, A @ Sigma @ A.T
    L = c.factor_W(xp.asarray(W))
    x_c = _host(c.correct(xp.asarray(x_unc), xp.asarray(V), L))
    mu_c = _host(c.correct(xp.asarray(mu), xp.asarray(V), L))

    Sigma_c = Sigma - V @ np.linalg.solve(W, V.T)
    eig = np.linalg.eigvalsh(Sigma_c)
    eig = eig[eig > 1e-10]
    reference = (
        -(n - k) / 2 * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(eig))
        - 0.5 * (x_c - mu_c) @ np.linalg.pinv(Sigma_c, rcond=1e-10) @ (x_c - mu_c)
    )

    log_p_x = (
        -0.5 * n * np.log(2 * np.pi)
        + 0.5 * np.linalg.slogdet(Q)[1]
        - 0.5 * (x_c - mu) @ Q @ (x_c - mu)
    )
    constant = -0.5 * np.linalg.slogdet(A @ A.T)[1] + 0.5 * k * np.log(2 * np.pi)
    got = log_p_x + constant + float(c.log_correction(L, xp.asarray(A @ x_c - A @ mu)))
    np.testing.assert_allclose(got, reference, rtol=1e-10)
