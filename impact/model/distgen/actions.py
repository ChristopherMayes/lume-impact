from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, model_validator

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract bases
# ------------------------------------------------------------------


class Action(BaseModel):
    """Base for read-only distgen actions.

    Subclasses must implement ``get``.  The ``model_validator`` enforces
    that the associated variable is marked read-only at construction time.
    """

    var: Variable

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def read_only(self) -> bool:
        return getattr(self.var, "read_only", False)

    @model_validator(mode="after")
    def _check_var(self) -> "Action":
        if not self.var.read_only:
            raise ValueError(f"{type(self).__name__} requires a read-only variable")
        return self

    @abstractmethod
    def get(self, gen: Any) -> Any: ...


class WritableAction(Action):
    """Base for distgen actions that support both get and set.

    Overrides ``_check_var`` so writable variables are accepted.
    Subclasses must implement ``get`` and ``set``.

    The model calls ``safe_set``, which enforces the read-only guard before
    delegating to ``set``.
    """

    @model_validator(mode="after")
    def _check_var(self) -> "WritableAction":
        return self

    @abstractmethod
    def set(self, gen: Any, value: Any) -> None: ...

    def safe_set(self, gen: Any, value: Any) -> None:
        if self.var.read_only:
            raise TypeError(f"'{self.name}' is read-only")
        self.set(gen, value)


# ------------------------------------------------------------------
# Concrete actions
# ------------------------------------------------------------------


class DistgenInputAction(WritableAction):
    """Maps a distgen input parameter via a colon-separated key (e.g. ``r_dist:sigma_xy:value``).

    If ``has_units`` is True the parameter is a quantity dict; the ``:value``
    suffix is already included in ``key`` and only the magnitude is read/written.
    """

    key: str
    has_units: bool

    def get(self, gen: Any) -> Any:
        return gen[self.key]

    def set(self, gen: Any, value: Any) -> None:
        gen[self.key] = value
