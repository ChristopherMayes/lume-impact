from __future__ import annotations

from typing import Any

from beamphysics import ParticleGroup

from impact.impact import Impact
from impact.model.config import VariableMappingConfig, make_actions
from lume.actions import Action, ActionModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn


class LUMEImpactModel(InitialParticlesMixIn, FinalParticlesMixIn, ActionModel[Impact]):
    """
    LUMEModel using the actions framework wrapping an Impact-T simulator object.
    """

    def __init__(
        self,
        impact: Impact,
        actions: list[Action],
        dummy_run: bool = False,
    ):
        super().__init__(simulator=impact, action_variables=actions)
        self.dummy_run = dummy_run

    @property
    def initial_particles(self) -> ParticleGroup:
        """
        Expose the initial particles provided to simulation using `InitialParticlesMixIn` for use in `StagedModel`.

        Returns
        -------
        ParticleGroup
            The starting particles from Impact-T
        """
        return self.simulator.initial_particles

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        """
        Expose the initial particles provided to simulation using `InitialParticlesMixIn` for use in `StagedModel`.

        Parameters
        ----------
        val : ParticleGroup
            The starting particles provided to Impact-T
        """
        self.simulator.initial_particles = val

    @property
    def final_particles(self) -> ParticleGroup:
        """
        Expose the final particles provided to simulation using `FinalParticlesMixIn` for use in `StagedModel`.

        Returns
        -------
        ParticleGroup
            The final particles as annotated by Impact-T
        """
        return self.simulator.particles.get("final_particles")

    @classmethod
    def from_impact(
        cls,
        impact: Impact,
        config: VariableMappingConfig | None = None,
        **kwargs,
    ) -> LUMEImpactModel:
        """
        Generate class populated with variables from an existing Impact-T object. Variable inclusion and naming is configured
        through VariableMappingConfig.

        Parameters
        ----------
        impact : Impact
            Impact-T session with lattice loaded and already run (to populate stats data and particle groups)
        config : VariableMappingConfig, optional
            Variable creation config. Defines mapping from Impact-T elements, attributes to variables, by
            default VariableMappingConfig()

        Returns
        -------
        LUMEImpactModel
            The generated model with action variables registered
        """
        return cls(impact, make_actions(impact, config), **kwargs)

    def _set(self, values: dict[str, Any]) -> None:
        super()._set(values)
        if not self.dummy_run:
            self.simulator.run()
