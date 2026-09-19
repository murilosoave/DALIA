# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from scipy.sparse import issparse


class LinearConstraintConfig(BaseModel):
    """Linear equality constraint ``A x = e`` on a block of latent parameters.

    ``type="linear"`` requires ``A`` with shape ``(k, n)`` and ``e`` with shape ``(k,)``.
    ``type="sum_to_zero"`` takes no data and expands to ``A = [1, ..., 1]``, ``e = [0]``
    once the size ``n`` of the constrained block is known (see ``LinearConstraint.from_config``).

    ``A`` and ``e`` are host-side objects (lists, numpy arrays or scipy sparse matrices);
    they are moved to the active array backend by ``LinearConstraint``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: Literal["linear", "sum_to_zero"] = "linear"
    A: Any = None
    e: Any = None

    @field_validator("A", mode="before")
    @classmethod
    def _coerce_A(cls, value):
        if value is None or issparse(value):
            return value
        return np.atleast_2d(np.asarray(value, dtype=float))

    @field_validator("e", mode="before")
    @classmethod
    def _coerce_e(cls, value):
        if value is None:
            return value
        return np.asarray(value, dtype=float).ravel()

    @model_validator(mode="after")
    def _check_shapes(self):
        if self.type == "sum_to_zero":
            if self.A is not None or self.e is not None:
                raise ValueError("'sum_to_zero' constraints take no 'A' or 'e'.")
            return self
        if self.A is None or self.e is None:
            raise ValueError("'linear' constraints require both 'A' and 'e'.")
        if self.A.ndim != 2:
            raise ValueError(f"'A' must be 2-D, got shape {self.A.shape}.")
        if self.A.shape[0] != self.e.shape[0]:
            raise ValueError(
                f"'A' has {self.A.shape[0]} rows but 'e' has {self.e.shape[0]} entries."
            )
        return self


def parse_config(config: dict | LinearConstraintConfig) -> LinearConstraintConfig:
    if isinstance(config, LinearConstraintConfig):
        return config
    return LinearConstraintConfig(**config)
