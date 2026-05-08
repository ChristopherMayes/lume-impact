from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Self

from pydantic import BaseModel, model_validator

from lume.variables import Variable
from impact.model.exceptions import ReadOnlyError

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
        return getattr(self.var, "read_only")

    @model_validator(mode="after")
    def _check_var(self) -> "Action[SimT]":
        if not self.var.read_only:
            raise ReadOnlyError(f"{type(self).__name__} requires a read-only variable")
        return self

    @abstractmethod
    def get(self, simulator: SimT) -> Any: ...


class WritableAction(Action[SimT], Generic[SimT]):
    """Base for actions that support both get and set.

    Overrides ``_check_var`` so writable variables are accepted.
    Subclasses must implement ``get`` and ``set``.

    The model calls ``set``, which enforces the read-only guard before
    delegating to ``_set``.
    """

    @model_validator(mode="after")
    def _check_var(self) -> Self:
        return self

    @abstractmethod
    def _set(self, simulator: SimT, value: Any) -> None:
        """
        User implentation for action goes here.

        Parameters
        ----------
        simulator: SimT
            The simulator object
        value: Any
            The value the variable associated with the action is being set to
        """
        ...

    def set(self, simulator: SimT, value: Any) -> None:
        """
        Outside facing set method with read-only checking.
    
        Parameters
        ----------
        simulator: SimT
            The simulator object
        value: Any
            The value the variable associated with the action is being set to
        """
        if self.var.read_only:
            raise ReadOnlyError(f"'{self.name}' is read-only")
        self._set(simulator, value)
