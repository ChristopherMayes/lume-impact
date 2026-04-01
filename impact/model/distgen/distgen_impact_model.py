from typing import Any

from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.actions import Action as ImpactAction
from impact.model.actions import WritableAction as WritableImpactAction
from impact.model.config import VariableMappingConfig
from impact.model.config import make_actions as make_impact_variables
from impact.model.distgen.actions import Action as DistgenAction
from impact.model.distgen.actions import WritableAction as WritableDistgenAction
from impact.model.distgen.config import DistgenVariableMappingConfig
from impact.model.distgen.config import make_actions as make_distgen_variables


class LUMEDistgenImpactModel(LUMEModel):
    """Combined distgen + Impact-T model.

    On each ``set`` call:
    1. Distgen inputs are written to ``gen`` and distgen is run.
    2. The resulting particle group is set as ``impact.initial_particles``.
    3. Impact inputs are written to ``impact`` and Impact is run.

    Variables are routed automatically based on which mapping list each
    variable name belongs to.
    """

    def __init__(
        self,
        gen: Any,
        impact: Impact,
        distgen_actions: list[DistgenAction],
        impact_actions: list[ImpactAction],
        dummy_run: bool = False,
    ):
        self.gen = gen
        self.impact = impact
        self.distgen_actions = distgen_actions
        self.impact_actions = impact_actions
        self._distgen_by_name: dict[str, DistgenAction] = {
            m.name: m for m in distgen_actions
        }
        self._impact_by_name: dict[str, ImpactAction] = {
            m.name: m for m in impact_actions
        }
        self.dummy_run = dummy_run
        self._state: dict[str, Any] = {}
        self.update_state()

    @classmethod
    def from_objects(
        cls,
        gen: Any,
        impact: Impact,
        distgen_config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
        impact_config: VariableMappingConfig = VariableMappingConfig(),
        **kwargs,
    ) -> "LUMEDistgenImpactModel":
        return cls(
            gen,
            impact,
            make_distgen_variables(gen, distgen_config),
            make_impact_variables(impact, impact_config),
            **kwargs,
        )

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {
            **{m.name: m.var for m in self.distgen_actions},
            **{m.name: m.var for m in self.impact_actions},
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        distgen_values = {
            name: value
            for name, value in values.items()
            if name in self._distgen_by_name
        }
        impact_values = {
            name: value
            for name, value in values.items()
            if name in self._impact_by_name
        }

        # Write distgen inputs and run distgen
        for name, value in distgen_values.items():
            action = self._distgen_by_name[name]
            if not isinstance(action, WritableDistgenAction):
                raise TypeError(f"'{action.name}' is read-only")
            action.safe_set(self.gen, value)

        if not self.dummy_run:
            self.gen.run()
            self.impact.initial_particles = self.gen.particles

        # Write impact inputs and run impact
        for name, value in impact_values.items():
            action = self._impact_by_name[name]
            if not isinstance(action, WritableImpactAction):
                raise TypeError(f"'{action.name}' is read-only")
            action.safe_set(self.impact, value)

        if not self.dummy_run:
            self.impact.run()

        self.update_state()

    def update_state(self) -> None:
        for m in self.distgen_actions:
            self._state[m.name] = m.get(self.gen)
        for m in self.impact_actions:
            self._state[m.name] = m.get(self.impact)

    def register_action(self, action: DistgenAction | ImpactAction) -> None:
        """Add a user-defined mapping to the model.

        Routes to the distgen or impact side based on the mapping's type.
        Replaces any existing mapping with the same name.
        """
        if isinstance(action, DistgenAction):
            if action.name in self._distgen_by_name:
                self.distgen_actions[
                    self.distgen_actions.index(self._distgen_by_name[action.name])
                ] = action
            else:
                self.distgen_actions.append(action)
            self._distgen_by_name[action.name] = action
            self._state[action.name] = action.get(self.gen)
        elif isinstance(action, ImpactAction):
            if action.name in self._impact_by_name:
                self.impact_actions[
                    self.impact_actions.index(self._impact_by_name[action.name])
                ] = action
            else:
                self.impact_actions.append(action)
            self._impact_by_name[action.name] = action
            self._state[action.name] = action.get(self.impact)
        else:
            raise TypeError(
                f"Expected DistgenAction or ImpactAction, got {type(action)}"
            )

    def reset(self) -> None:
        self.set(
            {
                m.name: m.var.default_value
                for m in (*self.distgen_actions, *self.impact_actions)
                if not m.read_only and hasattr(m.var, "default_value")
            }
        )
