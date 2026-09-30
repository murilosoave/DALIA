# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dalia.__init__ import NDArray, xp


class ConstraintConfig(BaseModel, ABC):
    """Linear constraint C x = e, declared on a submodel (acting on its block of
    the latent field) or on the model (acting on the whole latent field)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sum_to_zero", "linear"] = None

    @abstractmethod
    def build(self, n: int) -> tuple[NDArray, NDArray]:
        """Constraint rows C of shape (k, n) and right-hand side e of shape (k,)."""
        ...


class SumToZeroConstraintConfig(ConstraintConfig):
    """The n entries sum to zero."""

    def build(self, n: int) -> tuple[NDArray, NDArray]:
        return xp.ones((1, n), dtype=xp.float64), xp.zeros(1, dtype=xp.float64)


class LinearConstraintConfig(ConstraintConfig):
    """General constraint C x = e, e defaults to zero."""

    C: list[list[float]] = None
    e: list[float] = None

    @model_validator(mode="after")
    def check_shapes(self):
        if not self.C:
            raise ValueError("C must have at least one row.")
        if len({len(row) for row in self.C}) != 1:
            raise ValueError("All rows of C must have the same length.")
        if self.e is not None and len(self.e) != len(self.C):
            raise ValueError(
                f"e must have one entry per row of C ({len(self.C)} rows), got {len(self.e)}."
            )
        return self

    def build(self, n: int) -> tuple[NDArray, NDArray]:
        C = xp.asarray(self.C, dtype=xp.float64)
        if C.shape[1] != n:
            raise ValueError(f"C must have {n} columns, got {C.shape[1]}.")

        if self.e is None:
            e = xp.zeros(C.shape[0], dtype=xp.float64)
        else:
            e = xp.asarray(self.e, dtype=xp.float64)

        return C, e


def parse_config(config: dict | str) -> ConstraintConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    constraint_type = config.get("type")
    if constraint_type == "sum_to_zero":
        return SumToZeroConstraintConfig(**config)
    if constraint_type == "linear":
        return LinearConstraintConfig(**config)
    raise ValueError(f"Unknown constraint type: {constraint_type}")
