from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable, Variable

from impact.model.impact_config import (
    AttributeConfig,
    HeaderConfig,
    RunInfoConfig,
    StatAttributeConfig,
    StatsConfig,
    VariableMappingConfig,
)


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class ImpactVariableMapping(ABC):
    """Abstract base for variable mappings with integrated get/set logic.

    Each concrete subclass represents one variable and knows how to read
    and write its value directly from/to an Impact object.

    Subclasses define ``var`` as a Pydantic field and must implement ``get``.
    Writable subclasses should also override ``set``.
    """

    var: Variable  # defined as a Pydantic field in each concrete subclass

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def read_only(self) -> bool:
        return getattr(self.var, "read_only", False)

    @abstractmethod
    def get(self, imp: Any) -> Any:
        """Return the current value of this variable from *imp*."""

    def set(self, imp: Any, value: Any) -> None:
        """Write *value* to this variable on *imp*.

        Raises ``TypeError`` by default; override in writable subclasses.
        """
        raise TypeError(f"'{self.name}' is read-only")


# ------------------------------------------------------------------
# Concrete mapping types
# ------------------------------------------------------------------


class EleVariableMapping(ImpactVariableMapping, BaseModel):
    """Maps an element attribute: ``imp.ele[tool_name][tool_attrib]``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    control_name: str
    tool_name: str
    control_attrib: str
    tool_attrib: str
    var: ScalarVariable

    def get(self, imp: Any) -> Any:
        return imp.ele[self.tool_name][self.tool_attrib]

    def set(self, imp: Any, value: Any) -> None:
        imp.ele[self.tool_name][self.tool_attrib] = value


class HeaderVariableMapping(ImpactVariableMapping, BaseModel):
    """Maps a header key: ``imp.header[key]``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    var: ScalarVariable

    def get(self, imp: Any) -> Any:
        return imp.header[self.key]

    def set(self, imp: Any, value: Any) -> None:
        imp.header[self.key] = value


class StatVariableMapping(ImpactVariableMapping, BaseModel):
    """Maps an output stat: ``imp.stat(stat_name)``. Read-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stat_name: str
    var: NDVariable

    def get(self, imp: Any) -> Any:
        return imp.stat(self.stat_name)


class RunInfoVariableMapping(ImpactVariableMapping, BaseModel):
    """Maps a run_info entry: ``imp.output['run_info'][key]``. Read-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    var: ScalarVariable

    def get(self, imp: Any) -> Any:
        return imp.output["run_info"][self.key]


class ParticleGroupVariableMapping(ImpactVariableMapping, BaseModel):
    """Maps a particle group: ``imp.particles[tool_name]``.

    Only ``initial_particles`` is writable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    var: ParticleGroupVariable

    def get(self, imp: Any) -> Any:
        return imp.particles[self.tool_name]

    def set(self, imp: Any, value: Any) -> None:
        if self.tool_name == "initial_particles":
            imp.initial_particles = value
        else:
            raise TypeError(f"'{self.name}' is read-only")


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def make_mappings(
    imp: Any, config: VariableMappingConfig
) -> list[ImpactVariableMapping]:
    """Build variable mappings for every element attribute, header key, and output
    described by *config*.

    The current value in *imp* is used as ``default_value`` for each variable.
    """
    mappings: list[ImpactVariableMapping] = []

    if config.header is not None:
        for field_name, field_info in HeaderConfig.model_fields.items():
            attr_cfg = getattr(config.header, field_name)
            if not isinstance(attr_cfg, AttributeConfig):
                continue

            header_key = (
                field_info.alias if field_info.alias is not None else field_name
            )
            key_token = attr_cfg.alias if attr_cfg.alias is not None else header_key
            variable_name = config.header.pattern.format(key=key_token)

            mappings.append(
                HeaderVariableMapping(
                    key=header_key,
                    var=ScalarVariable(
                        name=variable_name,
                        default_value=imp.header.get(header_key),
                        unit=attr_cfg.unit,
                        read_only=attr_cfg.read_only,
                    ),
                )
            )

    if config.elements is not None:
        ele_name_map = (
            {v: k for k, v in config.elements.name_mappings.items()}
            if config.elements.name_mappings
            else None
        )
        ele_type_map = (
            {v: k for k, v in config.elements.type_mappings.items()}
            if config.elements.type_mappings
            else None
        )

        for ele in imp.lattice:
            ele_type: str = ele.get("type", "")
            ele_name: str = ele.get("name", "")

            type_cfg = getattr(config.elements, ele_type, None)
            if not isinstance(type_cfg, BaseModel):
                continue

            name_token = (
                ele_name_map.get(ele_name, ele_name) if ele_name_map else ele_name
            )
            type_token = (
                ele_type_map.get(ele_type, ele_type) if ele_type_map else ele_type
            )

            for field_name in type(type_cfg).model_fields:
                attr_cfg = getattr(type_cfg, field_name)
                if not isinstance(attr_cfg, AttributeConfig):
                    continue

                if field_name not in imp.ele[ele_name]:
                    continue

                attrib_token = (
                    attr_cfg.alias if attr_cfg.alias is not None else field_name
                )
                variable_name = config.elements.pattern.format(
                    type=type_token, name=name_token, attrib=attrib_token
                )

                mappings.append(
                    EleVariableMapping(
                        control_name=name_token,
                        tool_name=ele_name,
                        control_attrib=attrib_token,
                        tool_attrib=field_name,
                        var=ScalarVariable(
                            name=variable_name,
                            default_value=imp.ele[ele_name][field_name],
                            unit=attr_cfg.unit,
                            read_only=attr_cfg.read_only,
                        ),
                    )
                )

    if config.stats is not None:
        for field_name in StatsConfig.model_fields:
            stat_cfg = getattr(config.stats, field_name)
            if not isinstance(stat_cfg, StatAttributeConfig):
                continue
            name_token = stat_cfg.alias if stat_cfg.alias is not None else field_name
            variable_name = config.stats.pattern.format(name=name_token)
            stat_array = imp.stat(field_name)
            mappings.append(
                StatVariableMapping(
                    stat_name=field_name,
                    var=NDVariable(
                        name=variable_name,
                        shape=stat_array.shape,
                        default_value=stat_array,
                        unit=stat_cfg.unit,
                        read_only=True,
                    ),
                )
            )

    if config.run_info is not None:
        run_info_data = imp.output.get("run_info", {})
        for field_name in RunInfoConfig.model_fields:
            run_info_cfg = getattr(config.run_info, field_name)
            if not isinstance(run_info_cfg, StatAttributeConfig):
                continue
            key_token = (
                run_info_cfg.alias if run_info_cfg.alias is not None else field_name
            )
            variable_name = config.run_info.pattern.format(key=key_token)
            mappings.append(
                RunInfoVariableMapping(
                    key=field_name,
                    var=ScalarVariable(
                        name=variable_name,
                        default_value=run_info_data.get(field_name),
                        unit=run_info_cfg.unit,
                        read_only=True,
                    ),
                )
            )

    if config.particles is not None:
        reverse_particle_map = {v: k for k, v in config.particles.name_map.items()}
        particles_data = imp.output.get("particles", {})
        for tool_name in imp.particles.keys():
            control_name = reverse_particle_map.get(tool_name, tool_name)
            variable_name = config.particles.pattern.format(name=control_name)
            default_val = (
                getattr(imp, "initial_particles", None)
                if tool_name == "initial_particles"
                else particles_data.get(tool_name)
            )
            mappings.append(
                ParticleGroupVariableMapping(
                    tool_name=tool_name,
                    var=ParticleGroupVariable(
                        name=variable_name,
                        default_value=default_val,
                        read_only=tool_name != "initial_particles",
                    ),
                )
            )

    return mappings
