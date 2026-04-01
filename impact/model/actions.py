from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, model_validator

from lume.variables import Variable


# ------------------------------------------------------------------
# Abstract bases
# ------------------------------------------------------------------


class Action(ABC, BaseModel):
    """Base for read-only actions.

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
    def get(self, impact: Any) -> Any: ...


class WritableAction(Action, ABC):
    """Base for actions that support both get and set.

    Overrides ``_check_var`` so writable variables are accepted.
    Subclasses must implement ``get`` and ``set``.

    The model calls ``safe_set``, which enforces the read-only guard before
    delegating to ``set``.
    """

    @model_validator(mode="after")
    def _check_var(self) -> "WritableAction":
        return self

    @abstractmethod
    def set(self, impact: Any, value: Any) -> None: ...

    def safe_set(self, impact: Any, value: Any) -> None:
        if self.var.read_only:
            raise TypeError(f"'{self.name}' is read-only")
        self.set(impact, value)


# ------------------------------------------------------------------
# Concrete actions
# ------------------------------------------------------------------


class EleAction(WritableAction):
    """Maps an element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def get(self, impact: Any) -> Any:
        return impact.ele[self.ele_name][self.attribute]

    def set(self, impact: Any, value: Any) -> None:
        impact.ele[self.ele_name][self.attribute] = value


class HeaderAction(WritableAction):
    """Maps a header key: ``impact.header[key]``."""

    key: str

    def get(self, impact: Any) -> Any:
        return impact.header[self.key]

    def set(self, impact: Any, value: Any) -> None:
        impact.header[self.key] = value


class StatAction(Action):
    """Maps an output stat: ``impact.stat(stat_name)``. Read-only."""

    stat_name: str

    def get(self, impact: Any) -> Any:
        return impact.stat(self.stat_name)


class RunInfoAction(Action):
    """Maps a run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def get(self, impact: Any) -> Any:
        return impact.output["run_info"][self.key]


class ParticleGroupAction(WritableAction):
    """Maps a particle group: ``impact.particles[tool_name]``.

    Only ``initial_particles`` is writable; all other tool names must be
    constructed with ``var.read_only=True``.
    """

    tool_name: str

    @model_validator(mode="after")
    def _check_initial_particles(self) -> "ParticleGroupAction":
        if self.tool_name != "initial_particles" and not self.var.read_only:
            raise ValueError(
                f"Particle group '{self.tool_name}' is not writable; "
                "set var.read_only=True for non-initial_particles groups"
            )
        return self

    def get(self, impact: Any) -> Any:
        return impact.particles[self.tool_name]

    def set(self, impact: Any, value: Any) -> None:
        impact.initial_particles = value
