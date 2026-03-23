from typing import Any
from impact.impact import Impact
from lume.model import LUMEModel
from lume.variables import Variable

from impact.model.transformer import Transformer, RoutingImpactTransformer
from impact.model.config import HeaderConfig, VariableMappingConfig, make_variables


class LUMEImpactModel(LUMEModel):
    def __init__(
        self,
        imp: Impact,
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
        ele_pattern_override=None,
        ele_regex_override=None,
        header_pattern_override=None,
        header_regex_override=None,
        ele_name_mappings: dict[str, str] | None = None,
        ele_type_mappings: dict[str, str] | None = None,
        **kwargs,
    ):
        var_mappings = make_variables(
            imp,
            variable_mapping,
            ele_name_map={v: k for k, v in ele_name_mappings.items()}
            if ele_name_mappings
            else None,
            ele_type_map={v: k for k, v in ele_type_mappings.items()}
            if ele_type_mappings
            else None,
        )

        if transformer is None:
            # Build a map from attrib token -> actual imp.ele[name] key for any
            # element field where the AttributeConfig alias differs from the field name.
            attrib_map: dict[str, str] = {}
            ele_type_fields = {
                "drift",
                "quadrupole",
                "solenoid",
                "dipole",
                "solrf",
                "emfield_cartesian",
                "emfield_cylindrical",
            }
            for type_field in ele_type_fields:
                type_cfg = getattr(variable_mapping, type_field, None)
                if type_cfg is None:
                    continue
                for field_name in type_cfg.model_fields:
                    attr_cfg = getattr(type_cfg, field_name)
                    if attr_cfg is None or attr_cfg.alias is None:
                        continue
                    if attr_cfg.alias != field_name:
                        attrib_map[attr_cfg.alias] = field_name

            if (ele_pattern_override is not None) or (ele_regex_override is not None):
                _ele_pattern = ele_pattern_override
                _ele_regex = ele_regex_override
            else:
                _ele_pattern = variable_mapping.element_pattern
                _ele_regex = None

            _trans = RoutingImpactTransformer(
                ele_pattern=_ele_pattern,
                ele_regex=_ele_regex,
                ele_name_map=ele_name_mappings or {},
                ele_attrib_map=attrib_map or {},
            )

            # Build a map from variable name token -> actual imp.header key for any
            # header field where the AttributeConfig alias differs from the header key.
            key_map: dict[str, str] = {}
            if variable_mapping.header is not None:
                for field_name, field_info in HeaderConfig.model_fields.items():
                    attr_cfg = getattr(variable_mapping.header, field_name)
                    if attr_cfg is None:
                        continue
                    header_key = (
                        field_info.alias if field_info.alias is not None else field_name
                    )
                    key_token = (
                        attr_cfg.alias if attr_cfg.alias is not None else header_key
                    )
                    if key_token != header_key:
                        key_map[key_token] = header_key

            if (header_pattern_override is not None) or (
                header_regex_override is not None
            ):
                _header_pattern = header_pattern_override
                _header_regex = header_regex_override
            else:
                _header_pattern = variable_mapping.header_pattern
                _header_regex = None

            _trans.add_header_getter(
                pattern=_header_pattern, regex=_header_regex, key_map=key_map or None
            )
            _trans.add_header_setter(
                pattern=_header_pattern, regex=_header_regex, key_map=key_map or None
            )

        elif isinstance(transformer, Transformer):
            _trans = transformer

        else:
            raise ValueError(f"Unrecognized type for transformer: {type(transformer)}")

        # Get the vars
        vars = [x.var for x in var_mappings.header_mappings] + [
            x.var for x in var_mappings.ele_mappings
        ]

        # Construct model and check compatibility of variable names and transformer routes before sending out
        model = cls(imp, vars, _trans, **kwargs)
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
        raise NotImplementedError()
