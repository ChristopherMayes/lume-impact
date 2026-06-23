from __future__ import annotations

from distgen import Generator
from impact.impact import Impact
from lume.staged_model import StagedModel

from lume.actions import Action
from impact.model.config import VariableMappingConfig
from impact.model.distgen.config import DistgenVariableMappingConfig
from impact.model.distgen.model import LUMEDistgenModel
from impact.model.model import LUMEImpactModel


class LUMEDistgenImpactModel(StagedModel):
    """Combined distgen + Impact-T model using lume.StagedModel.

    ``LUMEDistgenModel`` and ``LUMEImpactModel`` are run in sequence:
    distgen runs first and its output particles are passed as initial particles
    to Impact before Impact runs.
    """

    def __init__(
        self,
        distgen_model: LUMEDistgenModel,
        impact_model: LUMEImpactModel,
    ):
        super().__init__([distgen_model, impact_model])

    @property
    def distgen_model(self) -> LUMEDistgenModel:
        return self.lume_model_instances[0]

    @property
    def impact_model(self) -> LUMEImpactModel:
        return self.lume_model_instances[1]

    @classmethod
    def from_objects(
        cls,
        gen: Generator,
        impact: Impact,
        distgen_config: DistgenVariableMappingConfig | None = None,
        impact_config: VariableMappingConfig | None = None,
        **kwargs,
    ) -> "LUMEDistgenImpactModel":
        distgen_model = LUMEDistgenModel.from_generator(
            gen, distgen_config or DistgenVariableMappingConfig(), **kwargs
        )
        impact_model = LUMEImpactModel.from_impact(
            impact, impact_config or VariableMappingConfig(), **kwargs
        )
        return cls(distgen_model, impact_model)

    def register_distgen_action_variable(self, action: Action) -> None:
        """Register an action variable on the distgen sub-model."""
        self.distgen_model.register_action_variable(action)

    def unregister_distgen_action_variable(self, name: str) -> None:
        """Unregister an action variable from the distgen sub-model by name."""
        self.distgen_model.unregister_action_variable(name)

    def register_impact_action_variable(self, action: Action) -> None:
        """Register an action variable on the impact sub-model."""
        self.impact_model.register_action_variable(action)

    def unregister_impact_action_variable(self, name: str) -> None:
        """Unregister an action variable from the impact sub-model by name."""
        self.impact_model.unregister_action_variable(name)
