# Copyright 2024-2025 DALIA authors. All rights reserved.

import math

from numba import njit, prange

from dalia import NDArray, sp, xp

# The row kernel is written once and compiled for the backend in use: the CPU
# (numpy) or, as a device function, the GPU (cupy).
if xp.__name__ == "cupy":
    from numba import cuda

    jit = cuda.jit(device=True)
else:
    jit = njit


@jit
def _row_variance(i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values):
    """sigma_i^2 = sum_{j,k} A_ij Sigma_jk A_ik over the nonzeros of row i of A.

    Sigma is symmetric, so only k >= j is visited (upper triangle of Sigma) and
    the off-diagonal terms count twice. The columns of A and of Sigma are sorted:
    Sigma_jk is reached by advancing in row j of Sigma from the last hit. Returns
    nan if Sigma_jk is not stored.
    """
    variance = 0.0
    for p in range(a_ptr[i], a_ptr[i + 1]):
        j = a_cols[p]
        lo, hi = s_ptr[j], s_ptr[j + 1]
        for q in range(p, a_ptr[i + 1]):
            k = a_cols[q]
            # advance in row j of Sigma up to column k
            while lo < hi and s_cols[lo] < k:
                lo += 1
            if lo == hi or s_cols[lo] != k:
                return math.nan
            # Sigma is symmetric: diagonal terms once, off-diagonal terms twice
            factor = 1.0 if p == q else 2.0
            variance += factor * a_values[p] * s_values[lo] * a_values[q]
    return variance


if xp.__name__ == "cupy":
    @cuda.jit
    def _variances(a_ptr, a_cols, a_values, s_ptr, s_cols, s_values, variances):
        i = cuda.grid(1)
        if i < variances.size:
            variances[i] = _row_variance(
                i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values
            )
else:
    @njit(parallel=True)
    def _variances(a_ptr, a_cols, a_values, s_ptr, s_cols, s_values, variances):
        for i in prange(variances.size):
            variances[i] = _row_variance(
                i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values
            )


def marginal_variances_linear_predictor(
    a: sp.sparse.csr_matrix, covariance: sp.sparse.csr_matrix
) -> NDArray:
    """Compute diag(A Sigma A^T) from selected entries of Sigma, without forming A Sigma.

    Parameters
    ----------
    a : sp.sparse.csr_matrix
        Design matrix A of shape (m, n), with sorted indices and no duplicates.
    covariance : sp.sparse.csr_matrix
        Upper triangle of the symmetric matrix Sigma of shape (n, n), with sorted
        indices. It must store Sigma_jk for every j <= k such that A_ij and A_ik
        are nonzeros of the same row i.

    Returns
    -------
    variances : NDArray
        Marginal variances of the linear predictor, of shape (m,).

    Notes
    -----
    Eq. C.0.3 of Gaedke-Merzhaeuser (2024), Appendix C:
    .. math:: sigma_i^2 = (A Sigma A^T)_ii = sum_k sum_j A_ij Sigma_jk A_ik
    restricted to the nonzeros of row i of A. The work scales with the number
    of nonzeros of A and the only array allocated is the result.
    """
    variances = xp.empty(a.shape[0], dtype=xp.float64)
    args = (
        a.indptr,
        a.indices,
        a.data,
        covariance.indptr,
        covariance.indices,
        covariance.data,
        variances,
    )

    if xp.__name__ == "cupy":
        # one GPU thread per row of A
        threads_per_block = 256
        blocks = (variances.size + threads_per_block - 1) // threads_per_block
        _variances[blocks, threads_per_block](*args)
    else:
        _variances(*args)

    if bool(xp.isnan(variances).any()):
        raise ValueError(
            "nan in the marginal variances: an entry of the covariance required "
            "by a row of A is not stored, or the inputs contain nan."
        )

    return variances
