from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, model_validator

from lume.variables import Variable

SimT = TypeVar("SimT")


class Action(ABC, BaseModel, Generic[SimT]):
    """Base for read-only actions over a generic simulator type.

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
    def _check_var(self) -> "Action[SimT]":
        if not self.var.read_only:
            raise ValueError(f"{type(self).__name__} requires a read-only variable")
        return self

    @abstractmethod
    def get(self, simulator: SimT) -> Any: ...


class WritableAction(Action[SimT], Generic[SimT]):
    """Base for actions that support both get and set.

    Overrides ``_check_var`` so writable variables are accepted.
    Subclasses must implement ``get`` and ``set``.

    The model calls ``safe_set``, which enforces the read-only guard before
    delegating to ``set``.
    """

    @model_validator(mode="after")
    def _check_var(self) -> "WritableAction[SimT]":
        return self

    @abstractmethod
    def set(self, simulator: SimT, value: Any) -> None: ...

    def safe_set(self, simulator: SimT, value: Any) -> None:
        if self.var.read_only:
            raise TypeError(f"'{self.name}' is read-only")
        self.set(simulator, value)
