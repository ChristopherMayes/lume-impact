from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class ImpactVarAction(ABC, BaseModel):
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

    @abstractmethod
    def get(self, imp: Any) -> Any:
        """Return the current value of this variable from Impact"""

    def set(self, imp: Any, value: Any) -> None:
        """Write the provided value to the Impact object."""
        raise TypeError(f"'{self.name}' is read-only")


# ------------------------------------------------------------------
# Concrete actions
# ------------------------------------------------------------------


class EleVarAction(ImpactVarAction):
    """Maps an element attribute: ``imp.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def get(self, imp: Any) -> Any:
        return imp.ele[self.ele_name][self.attribute]

    def set(self, imp: Any, value: Any) -> None:
        imp.ele[self.ele_name][self.attribute] = value


class HeaderVarAction(ImpactVarAction):
    """Maps a header key: ``imp.header[key]``."""

    key: str

    def get(self, imp: Any) -> Any:
        return imp.header[self.key]

    def set(self, imp: Any, value: Any) -> None:
        imp.header[self.key] = value


class StatVarAction(ImpactVarAction):
    """Maps an output stat: ``imp.stat(stat_name)``. Read-only."""

    stat_name: str

    def get(self, imp: Any) -> Any:
        return imp.stat(self.stat_name)


class RunInfoVarAction(ImpactVarAction):
    """Maps a run_info entry: ``imp.output['run_info'][key]``. Read-only."""

    key: str

    def get(self, imp: Any) -> Any:
        return imp.output["run_info"][self.key]


class ParticleGroupVarAction(ImpactVarAction):
    """Maps a particle group: ``imp.particles[tool_name]``.

    Only ``initial_particles`` is writable.
    """

    tool_name: str

    def get(self, imp: Any) -> Any:
        return imp.particles[self.tool_name]

    def set(self, imp: Any, value: Any) -> None:
        if self.tool_name == "initial_particles":
            imp.initial_particles = value
        else:
            raise TypeError(f"'{self.name}' is read-only")
