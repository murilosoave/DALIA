# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp
from dalia.configs.constraints_config import SumToZeroConstraintConfig
from dalia.configs.submodels_config import RW1SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RW1SubModel(SubModel):
    """Fit a random walk of order 1, x_{t+1} - x_t ~ N(0, 1 / tau).

    Its precision matrix is Q = tau * R, with structure matrix R = D^T D and D
    the matrix of first differences (Rue & Held 2005, Section 3.3.1). The model
    is intrinsic: R has rank n - 1, the constant vector is in its null space.
    """
    intrinsic: bool = True

    def __init__(
        self,
        config: RW1SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        n = self.n_latent_parameters
        if n < 2:
            raise ValueError(
                f"RW1 submodel requires at least 2 latent parameters, got {n}."
            )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix Q = tau * R."""

        # kwargs expects hyperparameters in external scale
        tau = kwargs.get("tau")
        n = self.n_latent_parameters

        # R = D^T D, D the matrix of first differences: tridiagonal (-1, 2, -1),
        # with 1 in the first and last diagonal entries
        diag = [2 * tau] * n
        diag[0] = diag[-1] = tau
        off_diag = [-tau] * (n - 1)

        Q_prior = sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1])

        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

        return Q_prior.tocoo()

    def _construct_identifiability_constraints(self) -> list:
        """The null space of R is the constant vector, so the prior does not
        determine the level of the field: sum(x) = 0."""
        return [SumToZeroConstraintConfig()]

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
