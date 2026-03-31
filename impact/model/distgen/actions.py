from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class Action(BaseModel):
    """
    Object containing a LUME variable and the action it performs on a distgen Generator.
    """

    var: Variable

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def read_only(self) -> bool:
        return getattr(self.var, "read_only", False)

    @abstractmethod
    def get(self, gen: Any) -> Any:
        """Return the current value of this variable from the Generator."""

    def set(self, gen: Any, value: Any) -> None:
        """Write the provided value to the Generator."""
        raise TypeError(f"'{self.name}' is read-only")


# ------------------------------------------------------------------
# Concrete actions
# ------------------------------------------------------------------


class DistgenInputAction(Action):
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
