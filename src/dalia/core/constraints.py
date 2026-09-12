# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix as host_csr_matrix
from scipy.sparse import issparse as host_issparse

from dalia import NDArray, sp, xp
from dalia.configs.constraints_config import LinearConstraintConfig
from dalia.utils import get_host

# Relative tolerance below which the Gram matrix A A^T is considered rank deficient.
_RANK_RTOL = 1e-12


class LinearConstraint:
    """``k`` linear equality constraints ``A x = e`` on ``n`` latent parameters.

    Notes
    -----
    Conditioning ``x ~ N(mu, Q^{-1})`` on ``A x = e`` (Rue & Held, Sec. 2.3) only needs
    ``V = Q^{-1} A^T`` (n x k) and ``W = A V`` (k x k):

        mean        mu* = mu - V W^{-1} (A mu - e)
        covariance  Sigma* = Q^{-1} - V W^{-1} V^T
        density     log p(x | Ax=e) = log N(x; mu, Q^{-1}) - 0.5 log|A A^T| + (k/2) log(2 pi)
                                      + 0.5 log|W| + 0.5 (Ax - A mu)^T W^{-1} (Ax - A mu)

    The solves producing ``V`` need the factorized ``Q`` and are done by the caller
    (``DALIA._solve_constraints``); this class holds ``A`` and ``e``, fills solver
    right-hand sides from the sparse ``A^T``, and does the small dense algebra on
    ``V`` and the Cholesky factor ``L`` of ``W``. ``A`` is a CSR matrix and ``e`` a
    1-D array of the active array backend; ``labels`` names the origin of each row
    (used in validation messages).
    """

    def __init__(self, A, e, labels: list[str] | None = None) -> None:
        """Build from host-side data: ``A`` a (k, n) array-like or scipy sparse matrix,
        ``e`` a (k,) array-like."""
        if not host_issparse(A):
            A = host_csr_matrix(np.atleast_2d(np.asarray(A, dtype=float)))
        A = host_csr_matrix(A)
        e = np.asarray(e, dtype=float).ravel()
        if A.shape[0] != e.shape[0]:
            raise ValueError(f"A has {A.shape[0]} rows but e has {e.shape[0]} entries.")
        if not np.all(np.isfinite(A.data)) or not np.all(np.isfinite(e)):
            raise ValueError("Constraint entries must be finite.")
        A.eliminate_zeros()
        self._set_backend(sp.sparse.csr_matrix(A), xp.asarray(e), labels)

    def _set_backend(self, A, e: NDArray, labels: list[str] | None) -> None:
        self.A = sp.sparse.csr_matrix(A)
        self.e: NDArray = e
        self.labels: list[str] = (
            list(labels) if labels is not None else [f"row {i}" for i in range(self.A.shape[0])]
        )
        if len(self.labels) != self.k:
            raise ValueError("One label per constraint row is required.")
        # COO of A^T, used to refill dense (n, k) right-hand sides without a dense copy of A^T.
        AT = self.A.T.tocoo()
        self._AT_rows = AT.row
        self._AT_cols = AT.col
        self._AT_data = AT.data
        self._G_chol = None  # Cholesky factor of A A^T (lazy)

    @classmethod
    def _from_backend(cls, A, e: NDArray, labels: list[str]) -> "LinearConstraint":
        """Build from data already on the active backend (used by embed/stack/subset)."""
        obj = cls.__new__(cls)
        obj._set_backend(A, e, labels)
        return obj

    # --- Construction -----------------------------------------------------

    @property
    def k(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @classmethod
    def sum_to_zero(cls, n: int, label: str = "sum_to_zero") -> "LinearConstraint":
        """The constraint ``sum(x) = 0`` on ``n`` variables."""
        return cls(np.ones((1, n)), np.zeros(1), labels=[label])

    @classmethod
    def from_null_space(cls, N, label: str = "null_space") -> "LinearConstraint":
        """``N^T x = 0`` for a null-space basis ``N`` of shape ``(n, r)``."""
        N = np.asarray(N, dtype=float)
        if N.ndim != 2:
            raise ValueError(f"A null-space basis must be 2-D, got shape {N.shape}.")
        r = N.shape[1]
        return cls(N.T, np.zeros(r), labels=[f"{label} {j}" for j in range(r)])

    @classmethod
    def from_config(
        cls, config: LinearConstraintConfig, n: int, label: str = "config"
    ) -> "LinearConstraint":
        """Materialize a config for a block of ``n`` latent parameters."""
        if config.type == "sum_to_zero":
            return cls.sum_to_zero(n, label=f"{label} (sum_to_zero)")
        if config.A.shape[1] != n:
            raise ValueError(
                f"Constraint matrix A has {config.A.shape[1]} columns but the "
                f"constrained block has {n} latent parameters."
            )
        k = config.A.shape[0]
        return cls(config.A, config.e, labels=[f"{label} row {i}" for i in range(k)])

    @classmethod
    def stack(cls, constraints: list["LinearConstraint"]) -> "LinearConstraint":
        """Stack constraints on the same ``n`` variables into one (rows concatenated)."""
        if len(constraints) == 0:
            raise ValueError("Cannot stack an empty list of constraints.")
        A = sp.sparse.vstack([c.A for c in constraints], format="csr")
        e = xp.concatenate([c.e for c in constraints])
        labels = [label for c in constraints for label in c.labels]
        return cls._from_backend(A, e, labels)

    def embed(self, offset: int, n_total: int) -> "LinearConstraint":
        """Express this constraint on a vector of length ``n_total`` whose entries
        ``[offset, offset + n)`` are the constrained block (zero padding elsewhere)."""
        left = sp.sparse.csr_matrix((self.k, offset), dtype=xp.float64)
        right = sp.sparse.csr_matrix((self.k, n_total - offset - self.n), dtype=xp.float64)
        A = sp.sparse.hstack([left, self.A, right], format="csr")
        return LinearConstraint._from_backend(A, self.e, self.labels)

    @staticmethod
    def _host_index(rows) -> np.ndarray:
        """Row selector (list, boolean mask or index array, host or device) as host indices."""
        rows = np.asarray(rows if isinstance(rows, (list, tuple)) else get_host(rows))
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        return rows.astype(np.int64)

    def subset(self, rows) -> "LinearConstraint":
        """The constraint made of the given rows (list, boolean mask or index array)."""
        rows = self._host_index(rows)
        A = self.A[xp.asarray(rows)] if len(rows) > 0 else sp.sparse.csr_matrix((0, self.n))
        e = self.e[xp.asarray(rows)]
        return LinearConstraint._from_backend(A, e, [self.labels[i] for i in rows])

    def support_mask(self, start: int, stop: int) -> np.ndarray:
        """Host boolean mask (k,) of the rows with a non-zero on columns ``[start, stop)``."""
        cols = get_host(self._AT_rows)  # row index of A^T == column index of A
        rows = get_host(self._AT_cols)
        touching = rows[(cols >= start) & (cols < stop)]
        mask = np.zeros(self.k, dtype=bool)
        mask[touching] = True
        return mask

    # --- Validation ---------------------------------------------------------

    def validate(self) -> None:
        """Check the constraint set is well posed: finite, no zero rows, full row rank.

        Uses the ``k x k`` Gram matrix ``A A^T`` (never densifies ``A``). Dependent rows are
        reported with their labels.
        """
        row_norms = get_host(xp.sqrt(xp.asarray(self.A.multiply(self.A).sum(axis=1)).ravel()))
        zero_rows = np.flatnonzero(row_norms == 0.0)
        if zero_rows.size > 0:
            names = ", ".join(self.labels[i] for i in zero_rows)
            raise ValueError(f"Constraint rows with all-zero coefficients: {names}.")

        G = get_host((self.A @ self.A.T).toarray())
        # Scale to unit diagonal so that the rank test is independent of row scaling.
        D = 1.0 / np.sqrt(np.diag(G))
        G_scaled = D[:, None] * G * D[None, :]
        eigvals = np.linalg.eigvalsh(G_scaled)
        if eigvals[0] <= _RANK_RTOL * eigvals[-1]:
            dependent = self._dependent_rows(G_scaled)
            names = ", ".join(self.labels[i] for i in dependent)
            raise ValueError(
                "Constraint rows are linearly dependent (redundant or contradictory): "
                f"{names}. Remove the redundant rows; intrinsic submodels receive their "
                "null-space constraints automatically."
            )

    @staticmethod
    def _dependent_rows(G: np.ndarray) -> list[int]:
        """Indices of rows that are linear combinations of earlier rows (incremental rank)."""
        dependent = []
        kept: list[int] = []
        for i in range(G.shape[0]):
            idx = kept + [i]
            sub = G[np.ix_(idx, idx)]
            eigvals = np.linalg.eigvalsh(sub)
            if eigvals[0] <= _RANK_RTOL * eigvals[-1]:
                dependent.append(i)
            else:
                kept.append(i)
        return dependent

    # --- Products ---------------------------------------------------------

    def residual(self, x: NDArray) -> NDArray:
        """``A x - e``."""
        return self.A @ x - self.e

    def _gram_chol(self) -> NDArray:
        if self._G_chol is None:
            self._G_chol = xp.linalg.cholesky((self.A @ self.A.T).toarray())
        return self._G_chol

    def project(self, x: NDArray) -> NDArray:
        """Minimum-norm correction onto ``{A x = e}``: ``x - A^T (A A^T)^{-1} (A x - e)``.

        Needs no precision factorization; used to make a starting point feasible.
        """
        return x - self.A.T @ self.solve_spd(self._gram_chol(), self.residual(x))

    def feasible_point(self) -> NDArray:
        """Minimum-norm solution of ``A x = e`` (zero when ``e = 0``)."""
        return self.project(xp.zeros(self.n, dtype=xp.float64))

    def fill_rhs(self, buffer: NDArray, rows=None) -> None:
        """Write ``A^T`` (or the columns of ``A^T`` for the given rows of ``A``) into ``buffer``.

        ``buffer`` has shape ``(n, k)`` (or ``(n, len(rows))``) and is overwritten.
        """
        buffer[:] = 0.0
        if rows is None:
            buffer[self._AT_rows, self._AT_cols] = self._AT_data
            return
        rows = self._host_index(rows)
        position = -np.ones(self.k, dtype=np.int64)
        position[rows] = np.arange(len(rows))
        cols_host = get_host(self._AT_cols)
        keep = position[cols_host] >= 0
        buffer[
            self._AT_rows[xp.asarray(keep)],
            xp.asarray(position[cols_host[keep]]),
        ] = self._AT_data[xp.asarray(keep)]

    # --- Small dense algebra on W = A V ------------------------------------

    @staticmethod
    def factor_W(W: NDArray) -> NDArray:
        """Cholesky factor ``L`` (lower) of ``W = A Q^{-1} A^T``."""
        W = 0.5 * (W + W.T)
        try:
            return xp.linalg.cholesky(W)
        except Exception as error:  # numpy LinAlgError / cupy equivalents
            raise ValueError(
                "A Q^{-1} A^T is not positive definite: the constraints are numerically "
                "dependent for this precision matrix."
            ) from error

    @staticmethod
    def solve_spd(L: NDArray, r: NDArray) -> NDArray:
        """Solve ``L L^T y = r`` with two triangular solves (``r`` may have several columns)."""
        y = sp.linalg.solve_triangular(L, r, lower=True)
        return sp.linalg.solve_triangular(L, y, lower=True, trans="T")

    def correct(self, x: NDArray, V: NDArray, L: NDArray) -> NDArray:
        """Constrained mean/mode: ``x - V W^{-1} (A x - e)``."""
        return x - V @ self.solve_spd(L, self.residual(x))

    @staticmethod
    def log_correction(L: NDArray, r: NDArray) -> float:
        """``0.5 log|W| + 0.5 r^T W^{-1} r`` for ``r = A x - A mu`` and ``W = L L^T``.

        The constant ``-0.5 log|A A^T| + (k/2) log(2 pi)`` of the constrained density is
        omitted: it appears identically in the prior and conditional terms of the
        objective and cancels.
        """
        logdet_W = 2.0 * xp.sum(xp.log(xp.diag(L)))
        z = sp.linalg.solve_triangular(L, r, lower=True)
        return 0.5 * logdet_W + 0.5 * (z @ z)

    @staticmethod
    def variance_correction(V: NDArray, L: NDArray, chunk_rows: int = 1 << 18) -> NDArray:
        """``diag(V W^{-1} V^T)`` computed in row chunks; subtract from ``diag(Q^{-1})``.

        Work is ``O(n k^2)``; temporary memory is ``chunk_rows x k``.
        """
        n = V.shape[0]
        out = xp.empty(n, dtype=xp.float64)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            Z = sp.linalg.solve_triangular(L, V[start:stop].T, lower=True)  # (k, m)
            out[start:stop] = xp.sum(Z * Z, axis=0)
        return out
