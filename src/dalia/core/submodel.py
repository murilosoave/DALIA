# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.constraints_config import SumToZeroConstraintConfig
from dalia.configs.submodels_config import SubModelConfig


class SubModel(ABC):
    """Abstract core class for statistical models."""

    # An intrinsic submodel has a rank-deficient precision matrix (e.g. a random
    # walk). Setting intrinsic = True in a submodel is all it takes: a constant is
    # added to the diagonal of its construct_Q_prior and a sum-to-zero constraint
    # to its constraints.
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

    def __init_subclass__(cls) -> None:
        if cls.intrinsic:
            cls.construct_Q_prior = cls._add_diagonal(cls.construct_Q_prior)

    @staticmethod
    def _add_diagonal(construct_Q_prior):
        """Wrap the construct_Q_prior of an intrinsic submodel: its precision matrix is
        rank deficient, so the constant config.diagonal is added to its diagonal,

        Q = construct_Q_prior(...) + diagonal * I, with I the identity matrix.
        """

        def wrapper(self, **kwargs):
            Q_prior = construct_Q_prior(self, **kwargs)
            Q_prior = Q_prior + self.config.diagonal * sp.sparse.identity(
                self.n_latent_parameters
            )

            # need this -> otherwise there might be a sorting issue
            Q_prior = Q_prior.tocsr()
            Q_prior.sort_indices()

            return Q_prior.tocoo()

        return wrapper

    def construct_constraints(self) -> tuple:
        """Construct the linear constraints C x = e declared in the config of the
        submodel, on its own latent parameters. An intrinsic submodel gets a
        sum-to-zero constraint first: its prior does not determine the level of
        the field.

        Returns
        -------
        C : NDArray
            Constraint matrix of shape (k, n_latent_parameters), None without constraints.
        e : NDArray
            Right-hand side of shape (k,), None without constraints.
        """
        constraints = list(self.config.constraints)
        if self.intrinsic:
            constraints.insert(0, SumToZeroConstraintConfig())

        if len(constraints) == 0:
            return None, None

        blocks = [
            constraint.build(self.n_latent_parameters) for constraint in constraints
        ]

        C = xp.concatenate([C for C, _ in blocks], axis=0)
        e = xp.concatenate([e for _, e in blocks])

        return C, e

    @abstractmethod
    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""
        ...

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
