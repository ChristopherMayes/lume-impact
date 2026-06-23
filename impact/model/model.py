from __future__ import annotations

from typing import Any

from beamphysics import ParticleGroup
from impact.impact import Impact
from lume.actions import ActionModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn

from lume.actions import Action
from impact.model.config import VariableMappingConfig, make_actions


class LUMEImpactModel(InitialParticlesMixIn, FinalParticlesMixIn, ActionModel[Impact]):
    def __init__(
        self,
        impact: Impact,
        actions: list[Action],
        dummy_run: bool = False,
    ):
        super().__init__(simulator=impact, action_variables=actions)
        self.dummy_run = dummy_run

    @property
    def impact(self) -> Impact:
        return self.simulator

    @property
    def initial_particles(self) -> ParticleGroup:
        return self.simulator.initial_particles

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        self.simulator.initial_particles = val

    @property
    def final_particles(self) -> ParticleGroup:
        return self.simulator.particles.get("final_particles")

    @classmethod
    def from_impact(
        cls,
        impact: Impact,
        config: VariableMappingConfig = VariableMappingConfig(),
        **kwargs,
    ) -> "LUMEImpactModel":
        return cls(impact, make_actions(impact, config), **kwargs)

    def _set(self, values: dict[str, Any]) -> None:
        super()._set(values)
        if not self.dummy_run:
            self.simulator.run()
