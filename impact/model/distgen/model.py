from typing import Any

from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.distgen.actions import Action
from impact.model.distgen.config import DistgenVariableMappingConfig, make_actions


class LUMEDistgenModel(LUMEModel):
    def __init__(
        self,
        gen: Any,
        actions: list[Action],
        dummy_run: bool = False,
    ):
        self.gen = gen
        self.actions = actions
        self._action_by_name: dict[str, Action] = {m.name: m for m in actions}
        self.dummy_run = dummy_run
        self._state: dict[str, Any] = {}
        self.update_state()

    @classmethod
    def from_generator(
        cls,
        gen: Any,
        config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
        **kwargs,
    ) -> "LUMEDistgenModel":
        return cls(gen, make_actions(gen, config), **kwargs)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {m.name: m.var for m in self.actions}

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self._action_by_name[name].set(self.gen, value)
        if not self.dummy_run:
            self.gen.run()
        self.update_state()

    def update_state(self) -> None:
        for m in self.actions:
            self._state[m.name] = m.get(self.gen)

    def register_action(self, action: Action) -> None:
        """Add a user-defined action to the model.

        The action's current value is read from ``gen`` immediately and
        stored in the state. If an action with the same name already exists
        it is replaced.
        """
        name = action.name
        if name in self._action_by_name:
            self.actions[self.actions.index(self._action_by_name[name])] = action
        else:
            self.actions.append(action)
        self._action_by_name[name] = action
        self._state[name] = action.get(self.gen)

    def reset(self) -> None:
        self.set(
            {
                m.name: m.var.default_value
                for m in self.actions
                if not m.read_only and hasattr(m.var, "default_value")
            }
        )
