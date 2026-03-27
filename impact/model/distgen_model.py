from typing import Any

from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.distgen_actions import DistgenVarAction
from impact.model.distgen_config import DistgenVariableMappingConfig, make_variables


class LUMEDistgenModel(LUMEModel):
    def __init__(
        self,
        gen: Any,
        mappings: list[DistgenVarAction],
        dummy_run: bool = False,
    ):
        self.gen = gen
        self.mappings = mappings
        self._mapping_by_name: dict[str, DistgenVarAction] = {
            m.name: m for m in mappings
        }
        self.dummy_run = dummy_run
        self._state: dict[str, Any] = {}
        self.update_state()

    @classmethod
    def from_generator(
        cls,
        gen: Any,
        variable_mapping: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
        **kwargs,
    ) -> "LUMEDistgenModel":
        return cls(gen, make_variables(gen, variable_mapping), **kwargs)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {m.name: m.var for m in self.mappings}

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self._mapping_by_name[name].set(self.gen, value)
        if not self.dummy_run:
            self.gen.run()
        self.update_state()

    def update_state(self) -> None:
        for m in self.mappings:
            self._state[m.name] = m.get(self.gen)

    def register_mapping(self, mapping: DistgenVarAction) -> None:
        """Add a user-defined mapping to the model.

        The mapping's current value is read from ``gen`` immediately and
        stored in the state. If a mapping with the same name already exists
        it is replaced.
        """
        name = mapping.name
        if name in self._mapping_by_name:
            self.mappings[self.mappings.index(self._mapping_by_name[name])] = mapping
        else:
            self.mappings.append(mapping)
        self._mapping_by_name[name] = mapping
        self._state[name] = mapping.get(self.gen)

    def reset(self) -> None:
        self.set(
            {
                m.name: m.var.default_value
                for m in self.mappings
                if not m.read_only and hasattr(m.var, "default_value")
            }
        )
