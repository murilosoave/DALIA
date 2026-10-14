# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.constraints_config import (
    LinearConstraintConfig,
    SumToZeroConstraintConfig,
)
from dalia.configs.submodels_config import RW2SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RW2SubModel(SubModel):
    """Fit a random walk of order 2, x_{t+2} - 2 x_{t+1} + x_t ~ N(0, 1 / tau).

    Its precision matrix is Q = tau * R, with structure matrix R = D^T D and D
    the matrix of second differences (Rue & Held 2005, Section 3.4.1). The model
    is intrinsic: R has rank n - 2, the constant and the linear trend are in its
    null space.
    """

    intrinsic: bool = True

    def __init__(
        self,
        config: RW2SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        n = self.n_latent_parameters
        if n < 3:
            raise ValueError(
                f"RW2 submodel requires at least 3 latent parameters, got {n}."
            )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix Q = tau * R."""

        # kwargs expects hyperparameters in external scale
        tau = kwargs.get("tau")
        n = self.n_latent_parameters

        # R = D^T D, D the (n - 2, n) matrix of second differences: pentadiagonal
        # with diagonal (1, 5, 6, ..., 6, 5, 1), first off-diagonal
        # (-2, -4, ..., -4, -2) and second off-diagonal 1
        diag = [6 * tau] * n
        diag[0] = diag[-1] = tau
        diag[1] = diag[-2] = 5 * tau
        off_diag_1 = [-4 * tau] * (n - 1)
        off_diag_1[0] = off_diag_1[-1] = -2 * tau
        off_diag_2 = [tau] * (n - 2)

        Q_prior = sp.sparse.diags(
            [off_diag_2, off_diag_1, diag, off_diag_1, off_diag_2],
            [-2, -1, 0, 1, 2],
        )

        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

        return Q_prior.tocoo()

    def _construct_identifiability_constraints(self) -> list:
        """The null space of R is spanned by the constant and the linear trend,
        so the prior determines neither the level nor the slope of the field:
        sum(x) = 0 and sum((t - mean(t)) x) = 0, t = 0, ..., n - 1. The trend
        row is centred to be orthogonal to the sum-to-zero row."""
        n = self.n_latent_parameters
        t = xp.arange(n, dtype=xp.float64)
        trend = t - t.mean()
        return [
            SumToZeroConstraintConfig(),
            LinearConstraintConfig(C=[trend.tolist()]),
        ]

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["tau", f"{self.config.tau:.3f}"],
            ["Diagonal", f"{self.config.diagonal:.1e}"],
        ]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )

        # Add the header title
        submodel_table = add_str_header(
            title=self.submodel_type.replace("_", " ").title(),
            table=submodel_table,
        )
        str_representation += submodel_table

        return str_representation
