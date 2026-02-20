from typing import Any
from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable
from distgen import Generator


class ImpactSimulator:
    def __init__(self, impact: Impact, beam_generator: Generator):
        self.impact = impact
        self.beam_generator = beam_generator

    def run(self) -> None:
        """
        Runs the Impact simulation with the current settings.

        This method should be called after setting all necessary properties on the Impact simulator.
        It will execute the simulation and update any relevant state or outputs as needed.
        """
        # generate beam distribution
        self.beam_generator.run()

        # set generated beam distribution to Impact initial particles
        self.impact.initial_particles = self.beam_generator.particles

        # run the Impact simulation
        self.impact.run()


class ImpactTransformer:
    def set_impact_property(
        self, simulator: ImpactSimulator, name: str, value: Any
    ) -> None:
        """
        Sets a property on the Impact simulator based on the variable name.

        Parameters
        ----------
        simulator : ImpactSimulator
            The Impact simulator instance to modify
        name : str
            The name of the variable to set
        value : Any
            The value to set for the specified variable
        """
        # Implement logic to map variable names to Impact properties and set them accordingly
        pass

    def get_impact_property(self, simulator: ImpactSimulator, name: str) -> Any:
        """
        Retrieves a property from the Impact simulator based on the variable name.

        Parameters
        ----------
        simulator : ImpactSimulator
            The Impact simulator instance to query
        name : str
            The name of the variable to retrieve

        Returns
        -------
        Any
            The current value of the specified variable from the simulator
        """
        # Implement logic to map variable names to Impact properties and retrieve their values accordingly
        pass


class LUMEImpactModel(LUMEModel):
    def __init__(
        self,
        simulator: ImpactSimulator,
        control_variables: dict[str, Variable],
        output_variables: dict[str, Variable],
        transformer: ImpactTransformer,
    ):
        self.simulator = simulator
        self.control_variables = control_variables
        self.output_variables = output_variables
        self.transformer = transformer

        self._state = {}
        self.update_state()

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
            self.transformer.set_impact_property(self.simulator, control_name, value)

        # run simulation
        self.simulator.run()

        # update state with new output values
        self.update_state()

    def update_state(self) -> None:
        """
        Internal method to update the model's state with current values from the simulator.

        This method retrieves the latest values for all control and output variables from the simulator
        and updates the internal state dictionary accordingly.
        """
        # Update state with current values from simulator
        for name in self.supported_variables.keys():
            self._state[name] = self.transformer.get_impact_property(
                self.simulator, name
            )
