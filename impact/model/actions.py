from typing import Any

from pydantic import model_validator

from impact.impact import Impact
from impact.model.base import Action, WritableAction


class ImpactAction(Action[Impact]):
    """Abstract base for all Impact actions."""


class WritableImpactAction(WritableAction[Impact], ImpactAction):
    """Abstract base for writable Impact actions."""


class EleAction(WritableImpactAction):
    """Maps an element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def get(self, impact: Impact) -> Any:
        return impact.ele[self.ele_name][self.attribute]

    def set(self, impact: Impact, value: Any) -> None:
        impact.ele[self.ele_name][self.attribute] = value


class HeaderAction(WritableImpactAction):
    """Maps a header key: ``impact.header[key]``."""

    key: str

    def get(self, impact: Impact) -> Any:
        return impact.header[self.key]

    def set(self, impact: Impact, value: Any) -> None:
        impact.header[self.key] = value


class StatAction(ImpactAction):
    """Maps an output stat: ``impact.stat(stat_name)``. Read-only."""

    stat_name: str

    def get(self, impact: Impact) -> Any:
        return impact.stat(self.stat_name)


class RunInfoAction(ImpactAction):
    """Maps a run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def get(self, impact: Impact) -> Any:
        return impact.output["run_info"][self.key]


class ParticleGroupAction(WritableImpactAction):
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

    def get(self, impact: Impact) -> Any:
        return impact.particles[self.tool_name]

    def set(self, impact: Impact, value: Any) -> None:
        impact.initial_particles = value
