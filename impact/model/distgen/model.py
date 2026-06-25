from __future__ import annotations

from typing import Any

from beamphysics import ParticleGroup
from distgen import Generator
from lume.actions import ActionModel
from lume.staged_model import FinalParticlesMixIn

from lume.actions import Action
from impact.model.distgen.config import DistgenVariableMappingConfig, make_actions


class LUMEDistgenModel(FinalParticlesMixIn, ActionModel[Generator]):
    def __init__(
        self,
        gen: Generator,
        actions: list[Action],
        dummy_run: bool = False,
    ):
        super().__init__(simulator=gen, action_variables=actions)
        self.dummy_run = dummy_run

    @property
    def final_particles(self) -> ParticleGroup:
        return self.simulator.particles

    @classmethod
    def from_generator(
        cls,
        gen: Generator,
        config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
        **kwargs,
    ) -> "LUMEDistgenModel":
        return cls(gen, make_actions(gen, config), **kwargs)

    def _set(self, values: dict[str, Any]) -> None:
        super()._set(values)
        if not self.dummy_run:
            self.simulator.run()
