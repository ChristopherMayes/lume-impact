from typing import Any

from distgen import Generator
from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.distgen_config import (
    DistgenVariableMappingConfig,
    make_transformer as make_distgen_transformer,
    make_variables as make_distgen_variables,
)
from impact.model.impact_config import (
    VariableMappingConfig,
    make_transformer as make_impact_transformer,
    make_variables as make_impact_variables,
)
from impact.model.impact_transformer import ImpactTransformer
from impact.model.transformer.base import Transformer
from impact.model.transformer.routing import RoutingTransformer


class LUMEImpactDistgenModel(LUMEModel):
    def __init__(
        self,
        imp: Impact,
        dg: Generator,
        vars: list[Variable],
        impact_transformer: ImpactTransformer,
        distgen_transformer: RoutingTransformer,
        impact_names: set[str],
        distgen_names: set[str],
        dummy_run: bool = False,
    ):
        self.imp = imp
        self.dg = dg
        self.vars = vars
        self.impact_transformer = impact_transformer
        self.distgen_transformer = distgen_transformer
        self.impact_names = impact_names
        self.distgen_names = distgen_names
        self.dummy_run = dummy_run

        self._state = {}
        self.update_state()

    @classmethod
    def from_config(
        cls,
        imp: Impact,
        dg: Generator,
        impact_config: VariableMappingConfig = VariableMappingConfig(),
        distgen_config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
        impact_transformer: ImpactTransformer | None = None,
        distgen_transformer: RoutingTransformer | None = None,
        **kwargs,
    ):
        imp_var_mappings = make_impact_variables(imp, impact_config)
        dg_var_mappings = make_distgen_variables(dg, distgen_config)

        if impact_transformer is None:
            _imp_trans = make_impact_transformer(impact_config)
        elif isinstance(impact_transformer, Transformer):
            _imp_trans = impact_transformer
        else:
            raise ValueError(
                f"Unrecognized type for impact_transformer: {type(impact_transformer)}"
            )

        if distgen_transformer is None:
            _dg_trans = make_distgen_transformer(dg_var_mappings, distgen_config)
        elif isinstance(distgen_transformer, Transformer):
            _dg_trans = distgen_transformer
        else:
            raise ValueError(
                f"Unrecognized type for distgen_transformer: {type(distgen_transformer)}"
            )

        all_vars = imp_var_mappings.all_vars + dg_var_mappings.all_vars
        impact_names = {v.name for v in imp_var_mappings.all_vars}
        distgen_names = {v.name for v in dg_var_mappings.all_vars}

        model = cls(
            imp,
            dg,
            all_vars,
            _imp_trans,
            _dg_trans,
            impact_names,
            distgen_names,
            **kwargs,
        )
        model.impact_transformer.check_ele_routes(imp, imp_var_mappings.ele_mappings)
        return model

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {var.name: var for var in self.vars}

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        distgen_values = {k: v for k, v in values.items() if k in self.distgen_names}
        impact_values = {k: v for k, v in values.items() if k in self.impact_names}

        for name, value in distgen_values.items():
            self.distgen_transformer.set_property(self.dg, name, value)

        for name, value in impact_values.items():
            self.impact_transformer.set_impact_property(self.imp, name, value)

        if not self.dummy_run:
            if distgen_values:
                self.dg.run()
                self.imp.initial_particles = self.dg.particles
            self.imp.run()

        self.update_state()

    def update_state(self) -> None:
        for name in self.impact_names:
            self._state[name] = self.impact_transformer.get_impact_property(
                self.imp, name
            )
        for name in self.distgen_names:
            self._state[name] = self.distgen_transformer.get_property(self.dg, name)

    def reset(self) -> None:
        self.set(
            {
                var.name: var.default_value
                for var in self.vars
                if not var.read_only and hasattr(var, "default_value")
            }
        )
