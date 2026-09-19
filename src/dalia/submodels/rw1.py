# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
from scipy.sparse import diags as host_diags
from tabulate import tabulate

from dalia import NDArray, sp
from dalia.configs.submodels_config import RW1SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RW1SubModel(SubModel):
    """Random walk of order 1 on a regular grid.

    Intrinsic GMRF with precision ``Q = tau * R``, ``R = D^T D`` for the first-difference
    matrix ``D``: ``R`` is tridiagonal with ``diag = [1, 2, ..., 2, 1]`` and ``-1`` off the
    diagonal, has rank ``n - 1`` and null space spanned by the constant vector. The model
    constrains the block on that null space automatically (``sum(x) = 0``) and evaluates
    its prior exactly on the constraint subspace (see ``SubModel.intrinsic``).
    """

    intrinsic = True

    def __init__(
        self,
        config: RW1SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        if self.n_latent_parameters < 2:
            raise ValueError("An RW1 submodel needs at least 2 latent parameters.")

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix ``tau * D^T D``."""
        tau = kwargs.get("tau")
        n = self.n_latent_parameters

        diag = np.full(n, 2.0 * tau)
        diag[0] = diag[-1] = tau
        off_diag = np.full(n - 1, -tau)

        Q_host = host_diags([off_diag, diag, off_diag], [-1, 0, 1], format="csr")
        Q_host.sort_indices()
        # Host first, then the active backend (cupyx accepts scipy matrices).
        return sp.sparse.csr_matrix(Q_host).tocoo()

    def null_space(self) -> NDArray:
        """The constant vector spans the null space of ``D^T D``."""
        return np.ones((self.n_latent_parameters, 1))

    def logdet_Q_prior_generalized(self, **kwargs) -> float:
        """``log|Q|^+ = (n - 1) log(tau) + log(n)``.

        The product of the non-zero eigenvalues of the path-graph Laplacian ``D^T D`` is
        ``n`` (Kirchhoff's theorem: ``n`` times the single spanning tree of a path), so
        the constant is included exactly.
        """
        n = self.n_latent_parameters
        return (n - 1) * np.log(kwargs.get("tau")) + np.log(n)

    def __str__(self) -> str:
        """String representation of the submodel."""
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["Rank deficiency", 1],
            ["tau", f"{self.config.tau:.3f}"],
        ]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )
        return add_str_header(
            title=self.submodel_type.replace("_", " ").upper(),
            table=submodel_table,
        )
