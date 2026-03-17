from typing import Any
from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.transformer import ImpactTransformer


class LUMEImpactModel(LUMEModel):
    def __init__(
        self,
        imp: Impact,
        control_variables: dict[str, Variable],
        output_variables: dict[str, Variable],
        transformer: ImpactTransformer,
    ):
        self.imp = imp
        self.control_variables = control_variables
        self.output_variables = output_variables
        self.transformer = transformer

        self._state = {}
        self.update_state()

    @classmethod
    def from_impact(self, imp: Impact):
        pass

    def supported_variables(self) -> dict[str, Variable]:
        """
        Returns a dictionary of all supported variables (both control and output) for this model.

        Returns
        -------
        dict[str, Variable]
            Dictionary mapping variable names to their corresponding Variable objects
        """
        return {**self.control_variables, **self.output_variables}

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
