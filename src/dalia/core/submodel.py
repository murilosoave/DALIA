# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.submodels_config import SubModelConfig
from dalia.core.constraints import LinearConstraint


class SubModel(ABC):
    """Abstract core class for statistical models."""

    # Intrinsic (improper) GMRFs have a singular prior precision (RW1, RW2, Besag, ...).
    # A subclass that sets ``intrinsic = True`` must implement ``null_space()`` and
    # ``logdet_Q_prior_generalized()``. The model then constrains the block on its full
    # null space (``N^T x = 0``) automatically and evaluates the prior as the proper
    # density restricted to that subspace: no regularization of Q_prior is needed.
    intrinsic: bool = False

    def __init__(
        self,
        config: SubModelConfig,
    ) -> None:
        """Initializes the model."""
        self.config = config
        self.input_path = Path(config.input_dir)
        self.submodel_type = config.type

        # --- Load design matrix

        try:
            a: spmatrix = load_npz(self.input_path.joinpath("a.npz"))
            self.a = sp.sparse.csc_matrix(a)
        except FileNotFoundError:
            # check if dense a matrix exists
            try:
                a: NDArray = np.load(self.input_path.joinpath("a.npy"))
                if xp == np:
                    self.a: NDArray = a
                else:
                    self.a: NDArray = xp.array(a)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No design matrix found. Please provide a valid design matrix."
                )

        self.n_latent_parameters: int = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            x_initial: NDArray = np.load(self.input_path.joinpath("x.npy"))
            if xp == np:
                self.x_initial: NDArray = x_initial
            else:
                self.x_initial: NDArray = xp.array(x_initial)
        except FileNotFoundError:
            self.x_initial: NDArray = xp.zeros((self.a.shape[1]), dtype=float)

        # --- Linear constraints A x = e on this block of latent parameters
        self.constraints: list[LinearConstraint] = [
            LinearConstraint.from_config(
                c, n=self.n_latent_parameters, label=f"{self.submodel_type} constraint {j}"
            )
            for j, c in enumerate(config.constraints)
        ]
        self.null_space_constraint: LinearConstraint | None = None
        if self.intrinsic:
            N = self.null_space()
            if N is None:
                raise NotImplementedError(
                    f"Intrinsic submodel {type(self).__name__} must implement null_space()."
                )
            N = np.asarray(N, dtype=float)
            if N.ndim != 2 or N.shape[0] != self.n_latent_parameters or N.shape[1] < 1:
                raise ValueError(
                    f"null_space() must return an (n, r) basis with n={self.n_latent_parameters}, "
                    f"got shape {N.shape}."
                )
            self.null_space_constraint = LinearConstraint.from_null_space(
                N, label=f"{self.submodel_type} null space"
            )


    @abstractmethod
    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""
        ...

    def null_space(self) -> NDArray | None:
        """Basis ``N`` (n, r) of the null space of the prior precision of an intrinsic
        submodel (e.g. a column of ones for RW1, ones and a linear trend for RW2).
        Proper submodels return None."""
        return None

    def logdet_Q_prior_generalized(self, **kwargs) -> float:
        """Generalized log-determinant of the singular prior precision of an intrinsic
        submodel (product of its non-zero eigenvalues), up to a theta-independent
        constant. For ``Q = tau R`` with rank deficiency ``r`` this is ``(n - r) log(tau)``."""
        raise NotImplementedError(
            f"{type(self).__name__} is not intrinsic or does not implement "
            "logdet_Q_prior_generalized()."
        )

    def load_a_predict(self) -> sp.sparse.csc_matrix:
        """Load the design matrix for prediction."""
        self.a_predict: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
            load_npz(self.input_path.joinpath("apr.npz"))
        )

        # check that number of columns is the same as in a
        if self.a_predict.shape[1] != self.a.shape[1]:
            raise ValueError(
                f"Number of columns in a_predict ({self.a_predict.shape[1]}) "
                f"does not match number of columns in a ({self.a.shape[1]})."
            )

        return self.a_predict
