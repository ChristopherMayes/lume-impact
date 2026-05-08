from typing import Any

from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.actions import ImpactAction, WritableImpactAction
from impact.model.config import VariableMappingConfig, make_actions
from impact.model.exceptions import ReadOnlyError


class LUMEImpactModel(LUMEModel):
    def __init__(
        self,
        impact: Impact,
        actions: list[ImpactAction],
        dummy_run: bool = False,
    ):
        self.impact = impact
        self.actions = actions
        self._action_by_name: dict[str, ImpactAction] = {m.name: m for m in actions}
        self.dummy_run = dummy_run
        self._state: dict[str, Any] = {}
        self.update_state()

    @classmethod
    def from_impact(
        cls,
        impact: Impact,
        config: VariableMappingConfig = VariableMappingConfig(),
        **kwargs,
    ) -> "LUMEImpactModel":
        return cls(impact, make_actions(impact, config), **kwargs)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {m.name: m.var for m in self.actions}

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        actions = {}
        for name, value in values.items():
            action = self._action_by_name[name]
            if not isinstance(action, WritableImpactAction):
                raise ReadOnlyError(f"'{action.name}' is read-only")
            actions[action] = value
        try:
            for action, value in actions.items():
                action.set(self.impact, value)
            if not self.dummy_run:
                self.impact.run()
        finally:
            self.update_state()

    def update_state(self) -> None:
        for m in self.actions:
            self._state[m.name] = m.get(self.impact)

    def register_action(self, action: ImpactAction) -> None:
        """Add a user-defined action to the model.

        The action's current value is read from ``impact`` immediately and
        stored in the state. If an action with the same name already exists
        it is replaced.
        """
        name = action.name
        if name in self._action_by_name:
            self.actions[self.actions.index(self._action_by_name[name])] = action
        else:
            self.actions.append(action)
        self._action_by_name[name] = action
        self._state[name] = action.get(self.impact)

    def unregister_action(self, name: str) -> None:
        """Remove an action from the model by name.

        Parameters
        ----------
        name : str
            Name of the action to remove.

        Raises
        ------
        KeyError
            If no action with the given name is registered.
        """
        if name not in self._action_by_name:
            raise KeyError(f"No action named '{name}' is registered")
        action = self._action_by_name.pop(name)
        self.actions.remove(action)
        self._state.pop(name, None)

    def reset(self) -> None:
        self.set(
            {
                m.name: m.var.default_value
                for m in self.actions
                if not m.read_only and hasattr(m.var, "default_value")
            }
        )
