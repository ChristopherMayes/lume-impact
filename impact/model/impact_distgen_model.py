from typing import Any
from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable
from distgen import Generator

from impact.model.transformer.transformer import Transformer
from impact.model.impact_config import (
    VariableMappingConfig,
    make_transformer,
    make_variables,
)


class LUMEImpactDistgenModel(LUMEModel):
    def __init__(
        self,
        imp: Impact,
        dg: Generator,
        vars: list[Variable],
        transformer: Transformer,
        dummy_run: bool = False,
    ):
        self.imp = imp
        self.vars = vars
        self.transformer = transformer
        self.dummy_run = dummy_run

        self._state = {}
        self.update_state()

    @classmethod
    def from_impact(
        cls,
        imp: Impact,
        variable_mapping: VariableMappingConfig = VariableMappingConfig(),
        transformer: Transformer | None = None,
        **kwargs,
    ):
        var_mappings = make_variables(imp, variable_mapping)

        if transformer is None:
            _trans = make_transformer(variable_mapping)
        elif isinstance(transformer, Transformer):
            _trans = transformer
        else:
            raise ValueError(f"Unrecognized type for transformer: {type(transformer)}")

        model = cls(imp, var_mappings.all_vars, _trans, **kwargs)
        model.transformer.check_ele_routes(imp, var_mappings.ele_mappings)
        return model

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """
        Returns a dictionary of all supported variables (both control and output) for this model.

        Returns
        -------
        dict[str, Variable]
            Dictionary mapping variable names to their corresponding Variable objects
        """
        return {var.name: var for var in self.vars}

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to retrieve current values for specified variables.

        Parameters
        ----------
        names : list[str]
            List of variable names to retrieve

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values
        """
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set input variables and compute outputs.

        This method:
        1. Updates input variables in the state
        2. Performs calculations to update output variables
        3. Stores results in the state

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set
        """

        # set PV values to simulation
        for control_name, value in values.items():
            self.transformer.set_impact_property(self.imp, control_name, value)

        # run simulation
        if not self.dummy_run:
            self.imp.run()

        # update state with new output values
        self.update_state()

    def update_state(self) -> None:
        """
        Internal method to update the model's state with current values from the simulator.

        This method retrieves the latest values for all control and output variables from the simulator
        and updates the internal state dictionary accordingly.
        """
        # Update state with current values from impact
        for name in self.supported_variables.keys():
            self._state[name] = self.transformer.get_impact_property(self.imp, name)

    def reset(self) -> None:
        self.set(
            {
                var.name: var.default_value
                for var in self.supported_variables
                if not var.read_only and hasattr(var, "default_value")
            }
        )
