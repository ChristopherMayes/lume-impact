from __future__ import annotations

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

    @classmethod
    def from_impact(
        cls,
        impact: Impact,
        config: VariableMappingConfig | None = None,
        **kwargs,
    ) -> "LUMEImpactModel":
        if config is None:
            config = VariableMappingConfig()
        return cls(impact, make_actions(impact, config), **kwargs)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {m.name: m.var for m in self.actions}

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._action_by_name[name].get(self.impact) for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        to_set: list[tuple[WritableImpactAction, Any]] = []
        for name, value in values.items():
            action = self._action_by_name[name]
            if not isinstance(action, WritableImpactAction):
                raise ReadOnlyError(f"'{action.name}' is read-only")
            to_set.append((action, value))
        for action, value in to_set:
            action._set(self.impact, value)
        if not self.dummy_run:
            self.impact.run()

    def register_action(self, action: ImpactAction) -> None:
        """Add a user-defined action to the model.

        If an action with the same name already exists it is replaced.
        """
        name = action.name
        if name in self._action_by_name:
            self.actions[self.actions.index(self._action_by_name[name])] = action
        else:
            self.actions.append(action)
        self._action_by_name[name] = action

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

    def reset(self) -> None:
        self.set(
            {
                m.name: m.var.default_value
                for m in self.actions
                if not m.read_only and hasattr(m.var, "default_value")
            }
        )
