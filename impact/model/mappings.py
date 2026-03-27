from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class ImpactVariableMapping(ABC, BaseModel):
    """Abstract base for variable mappings with integrated get/set logic.

    Each concrete subclass represents one variable and knows how to read
    and write its value directly from/to an Impact object.

    Subclasses define ``var`` as a Pydantic field and must implement ``get``.
    Writable subclasses should also override ``set``.
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
        """Return the current value of this variable from *imp*."""

    def set(self, imp: Any, value: Any) -> None:
        """Write *value* to this variable on *imp*.

        Raises ``TypeError`` by default; override in writable subclasses.
        """
        raise TypeError(f"'{self.name}' is read-only")


# ------------------------------------------------------------------
# Concrete mapping types
# ------------------------------------------------------------------


class EleVariableMapping(ImpactVariableMapping):
    """Maps an element attribute: ``imp.ele[tool_name][tool_attrib]``."""

    control_name: str
    tool_name: str
    control_attrib: str
    tool_attrib: str

    def get(self, imp: Any) -> Any:
        return imp.ele[self.tool_name][self.tool_attrib]

    def set(self, imp: Any, value: Any) -> None:
        imp.ele[self.tool_name][self.tool_attrib] = value


class HeaderVariableMapping(ImpactVariableMapping):
    """Maps a header key: ``imp.header[key]``."""

    key: str

    def get(self, imp: Any) -> Any:
        return imp.header[self.key]

    def set(self, imp: Any, value: Any) -> None:
        imp.header[self.key] = value


class StatVariableMapping(ImpactVariableMapping):
    """Maps an output stat: ``imp.stat(stat_name)``. Read-only."""

    stat_name: str

    def get(self, imp: Any) -> Any:
        return imp.stat(self.stat_name)


class RunInfoVariableMapping(ImpactVariableMapping):
    """Maps a run_info entry: ``imp.output['run_info'][key]``. Read-only."""

    key: str

    def get(self, imp: Any) -> Any:
        return imp.output["run_info"][self.key]


class ParticleGroupVariableMapping(ImpactVariableMapping):
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
