from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, model_validator

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class Action(ABC, BaseModel):
    """
    Object containing a LUME variable and the action it performs on a LUME `Impact` object.
    """

    var: Variable  # defined as a Pydantic field in each concrete subclass

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def read_only(self) -> bool:
        return getattr(self.var, "read_only", False)

    def get(self, impact: Any) -> Any:
        """User callable get. Implement _get in your subclass"""
        return self._get(impact)

    @abstractmethod
    def _get(self, impact: Any) -> Any:
        """Return the current value of this variable from Impact"""

    def set(self, impact: Any, value: Any) -> None:
        """User callable set function. Implement _set in your subclasses."""
        if self.var.read_only:
            raise TypeError(f"'{self.name}' is read-only")
        return self._set(impact, value)

    def _set(self, impact: Any, value: Any) -> None:
        """Set action associated with the variable. Don't need to implement if read-only."""
        raise NotImplementedError()


# ------------------------------------------------------------------
# Concrete actions
# ------------------------------------------------------------------


class EleAction(Action):
    """Maps an element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def _get(self, impact: Any) -> Any:
        return impact.ele[self.ele_name][self.attribute]

    def _set(self, impact: Any, value: Any) -> None:
        impact.ele[self.ele_name][self.attribute] = value


class HeaderAction(Action):
    """Maps a header key: ``impact.header[key]``."""

    key: str

    def _get(self, impact: Any) -> Any:
        return impact.header[self.key]

    def _set(self, impact: Any, value: Any) -> None:
        impact.header[self.key] = value


class StatAction(Action):
    """Maps an output stat: ``impact.stat(stat_name)``. Read-only."""

    stat_name: str

    def _get(self, impact: Any) -> Any:
        return impact.stat(self.stat_name)

    @model_validator(mode="after")
    def check_read_only(self) -> "RunInfoAction":
        if not self.var.read_only:
            raise ValueError("Variable must be read-only for stat action")
        return self


class RunInfoAction(Action):
    """Maps a run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, impact: Any) -> Any:
        return impact.output["run_info"][self.key]

    @model_validator(mode="after")
    def check_read_only(self) -> "RunInfoAction":
        if not self.var.read_only:
            raise ValueError("Variable must be read-only for run info action")
        return self


class ParticleGroupAction(Action):
    """Maps a particle group: ``impact.particles[tool_name]``.

    Only ``initial_particles`` is writable.
    """

    tool_name: str

    def _get(self, impact: Any) -> Any:
        return impact.particles[self.tool_name]

    def _set(self, impact: Any, value: Any) -> None:
        if self.tool_name == "initial_particles":
            impact.initial_particles = value
        else:
            raise TypeError(f"'{self.name}' is read-only")
