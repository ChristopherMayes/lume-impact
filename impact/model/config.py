from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable

from impact.impact import Impact, RUN_INFO_UNITS, STAT_UNITS
from impact.parsers import ELE_UNITS, HEADER_UNITS
from impact.model.actions import (
    Action,
    EleAction,
    HeaderAction,
    StatAction,
    RunInfoAction,
    ParticleGroupAction,
)

logger = logging.getLogger(__name__)


class AttributeConfig(BaseModel):
    """Config for a single header or element attribute.

    Units are looked up automatically from ``HEADER_UNITS`` or ``ELE_UNITS``.

    Parameters
    ----------
    read_only : bool, optional
        Whether the variable is read-only.  Defaults to ``False``.
    """

    read_only: bool = False


class ConfigBase(BaseModel):
    attrib_map: dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            default = field.default
            if not isinstance(default, BaseModel):
                continue
            key = (
                name
                if name in data
                else (field.alias if field.alias and field.alias in data else None)
            )
            if key is None:
                continue
            field_data = data[key]
            if isinstance(field_data, dict):
                for attr, val in default.model_dump().items():
                    field_data.setdefault(attr, val)
        return data


# ------------------------------------------------------------------
# Per-element-type configs
# Set a field to AttributeConfig() to include it; leave as None to exclude.
# ------------------------------------------------------------------


class DriftConfig(ConfigBase):
    zedge: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()


class QuadrupoleConfig(ConfigBase):
    b1_gradient: AttributeConfig | None = AttributeConfig()
    L_effective: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    rf_phase_deg: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()


class SolenoidConfig(ConfigBase):
    b_field: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()


class DipoleConfig(ConfigBase):
    b_field: AttributeConfig | None = AttributeConfig()
    b_field_x: AttributeConfig | None = AttributeConfig()
    half_gap: AttributeConfig | None = AttributeConfig()


class SolrfConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    solenoid_field_scale: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()


class EmfieldCartesianConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()


class EmfieldCylindricalConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()


# ------------------------------------------------------------------
# Header config
# ------------------------------------------------------------------


class HeaderConfig(ConfigBase):
    """
    Configuration for variables generated from Impact-T's header data.

    `pattern` is a python f-string which is used to set the variable name. The parameter `key` is passed to it representing the
    name of the value in Impact-T's header data.

    Units are sourced automatically from ``HEADER_UNITS`` in ``parsers.py``.
    Set a field to ``None`` to exclude it.
    """

    model_config = ConfigDict(populate_by_name=True)

    pattern: str = "header:{key}"

    # Processor domain
    Npcol: AttributeConfig | None = AttributeConfig()
    Nprow: AttributeConfig | None = AttributeConfig()
    # Time stepping
    Dt: AttributeConfig | None = AttributeConfig()
    Ntstep: AttributeConfig | None = AttributeConfig()
    # Beam / bunch
    Nbunch: AttributeConfig | None = AttributeConfig()
    Np: AttributeConfig | None = AttributeConfig()
    Bcurr: AttributeConfig | None = AttributeConfig()
    Bkenergy: AttributeConfig | None = AttributeConfig()
    Bmass: AttributeConfig | None = AttributeConfig()
    Bcharge: AttributeConfig | None = AttributeConfig()
    Bfreq: AttributeConfig | None = AttributeConfig()
    Tini: AttributeConfig | None = AttributeConfig()
    # Flags
    Flagmap: AttributeConfig | None = AttributeConfig()
    Flagerr: AttributeConfig | None = AttributeConfig()
    Flagdiag: AttributeConfig | None = AttributeConfig()
    Flagimg: AttributeConfig | None = AttributeConfig()
    Flagdist: AttributeConfig | None = AttributeConfig()
    Flagbc: AttributeConfig | None = AttributeConfig()
    Rstartflg: AttributeConfig | None = AttributeConfig()
    Flagsbstp: AttributeConfig | None = AttributeConfig()
    # Grid
    Dim: AttributeConfig | None = AttributeConfig()
    Nx: AttributeConfig | None = AttributeConfig()
    Ny: AttributeConfig | None = AttributeConfig()
    Nz: AttributeConfig | None = AttributeConfig()
    Xrad: AttributeConfig | None = AttributeConfig()
    Yrad: AttributeConfig | None = AttributeConfig()
    Perdlen: AttributeConfig | None = AttributeConfig()
    Zimage: AttributeConfig | None = AttributeConfig()
    # Emission
    Nemission: AttributeConfig | None = AttributeConfig()
    Temission: AttributeConfig | None = AttributeConfig()
    # Distribution parameters
    sigx: AttributeConfig | None = AttributeConfig()
    sigpx: AttributeConfig | None = AttributeConfig()
    muxpx: AttributeConfig | None = AttributeConfig()
    xscale: AttributeConfig | None = AttributeConfig()
    pxscale: AttributeConfig | None = AttributeConfig()
    xmu1: AttributeConfig | None = AttributeConfig()
    xmu2: AttributeConfig | None = AttributeConfig()
    sigy: AttributeConfig | None = AttributeConfig()
    sigpy: AttributeConfig | None = AttributeConfig()
    muxpy: AttributeConfig | None = AttributeConfig()
    yscale: AttributeConfig | None = AttributeConfig()
    pyscale: AttributeConfig | None = AttributeConfig()
    ymu1: AttributeConfig | None = AttributeConfig()
    ymu2: AttributeConfig | None = AttributeConfig()
    sigz: AttributeConfig | None = AttributeConfig()
    sigpz: AttributeConfig | None = AttributeConfig()
    muxpz: AttributeConfig | None = AttributeConfig()
    zscale: AttributeConfig | None = AttributeConfig()
    pzscale: AttributeConfig | None = AttributeConfig()
    zmu1: AttributeConfig | None = AttributeConfig()
    zmu2: AttributeConfig | None = AttributeConfig()


# ------------------------------------------------------------------
# Stats config
# ------------------------------------------------------------------


class StatsConfig(ConfigBase):
    """
    Configuration for making variables from Impact-T stats data.

    `pattern` is a python f-string and the value `name` is passed to it during formatting to generate variable names. `name`
    corresponds to the name of the value within Impact-T's stats data.

    `max_size` is the maximum size of the stats array. Arrays will be padded and trimmed to this length to fit the fixed-length
    variable. If unset, the length of stats arrays in the Impact-T object will be used.

    Each stat field is a ``bool``: ``True`` (default) includes the stat, ``False`` excludes it.
    Units are sourced automatically from ``STAT_UNITS`` in ``impact.py``.
    """

    pattern: str = "stat:{name}"
    max_size: int | None = None

    mean_kinetic_energy: bool = True
    mean_x: bool = True
    mean_y: bool = True
    mean_z: bool = True
    sigma_x: bool = True
    sigma_y: bool = True
    sigma_z: bool = True
    norm_emit_x: bool = True
    norm_emit_y: bool = True
    norm_emit_z: bool = True


# ------------------------------------------------------------------
# Run info config
# ------------------------------------------------------------------


class RunInfoConfig(ConfigBase):
    """
    Configuration for variables corresponding to "run info" within Impact-T.

    `pattern` is a python f-string where the value `key` corresponding to the run info key name in LUMEImpact is
    passed during formatting to make the variable name.

    Each field is a ``bool``: ``True`` (default) includes the entry, ``False`` excludes it.
    Units are sourced automatically from ``RUN_INFO_UNITS`` in ``impact.py``.
    """

    pattern: str = "run_info:{key}"

    run_time: bool = True
    error: bool = True


# ------------------------------------------------------------------
# Element container + top-level mapping config
# ------------------------------------------------------------------


class ElementsConfig(BaseModel):
    """Groups all per-element-type attribute configs and element routing settings.

    Set a type to ``None`` to skip all variables for that element type.

    `pattern` is a python f-string where the values `name` and `attrib` corresponding to the element name and name of the
    attribute are passed during formatting to generate the variable name.

    Attribute units are sourced automatically from ``ELE_UNITS`` in ``parsers.py``.
    """

    pattern: str = "ele:{name}:{attrib}"
    control_to_tool_name: dict[str, str] | None = None  # control_name -> tool_name
    control_to_tool_type: dict[str, str] | None = None  # control_type -> tool_type

    drift: DriftConfig | None = DriftConfig()
    quadrupole: QuadrupoleConfig | None = QuadrupoleConfig()
    solenoid: SolenoidConfig | None = SolenoidConfig()
    dipole: DipoleConfig | None = DipoleConfig()
    solrf: SolrfConfig | None = SolrfConfig()
    emfield_cartesian: EmfieldCartesianConfig | None = EmfieldCartesianConfig()
    emfield_cylindrical: EmfieldCylindricalConfig | None = EmfieldCylindricalConfig()


class ParticlesConfig(BaseModel):
    """
    Config for particle group variables.

    `pattern` is a python f-string where the value `name` correspond to the named particle group the variable exposes.
    """

    pattern: str = "particles:{name}"
    control_to_tool_name: dict[str, str] | None = None  # control_name -> tool_name

    @property
    def name_map(self) -> dict[str, str]:
        """Maps particle variable token (control name) -> actual key in impact.particles."""
        return self.control_to_tool_name or {}


class VariableMappingConfig(BaseModel):
    """Maps Impact-T element attributes, header keys, and output stats to model variable names.

    Set any sub-config to ``None`` to skip that category entirely.
    """

    header: HeaderConfig | None = HeaderConfig()
    elements: ElementsConfig | None = ElementsConfig()
    stats: StatsConfig | None = StatsConfig()
    run_info: RunInfoConfig | None = RunInfoConfig()
    particles: ParticlesConfig | None = ParticlesConfig()


def _make_header_actions(impact: Any, config: HeaderConfig) -> list[Action]:
    actions = []
    for field_name, field_info in HeaderConfig.model_fields.items():
        attr_cfg = getattr(config, field_name)
        if not isinstance(attr_cfg, AttributeConfig):
            continue
        header_key = field_info.alias if field_info.alias is not None else field_name
        key_token = config.attrib_map.get(header_key, header_key)
        actions.append(
            HeaderAction(
                key=header_key,
                var=ScalarVariable(
                    name=config.pattern.format(key=key_token),
                    default_value=impact.header.get(header_key),
                    unit=HEADER_UNITS.get(header_key),
                    read_only=attr_cfg.read_only,
                ),
            )
        )
    return actions


def _make_element_actions(impact: Any, config: ElementsConfig) -> list[Action]:
    actions = []
    ele_name_map = (
        {v: k for k, v in config.control_to_tool_name.items()}
        if config.control_to_tool_name
        else {}
    )
    ele_type_map = (
        {v: k for k, v in config.control_to_tool_type.items()}
        if config.control_to_tool_type
        else {}
    )
    for ele in impact.lattice:
        ele_type: str = ele.get("type", "")
        ele_name: str = ele.get("name", "")
        type_cfg = getattr(config, ele_type, None)
        if not isinstance(type_cfg, BaseModel):
            continue
        name_token = ele_name_map.get(ele_name, ele_name)
        type_token = ele_type_map.get(ele_type, ele_type)
        for field_name in type(type_cfg).model_fields:
            attr_cfg = getattr(type_cfg, field_name)
            if not isinstance(attr_cfg, AttributeConfig):
                continue
            if field_name not in impact.ele[ele_name]:
                continue
            attrib_token = type_cfg.attrib_map.get(field_name, field_name)
            actions.append(
                EleAction(
                    ele_name=ele_name,
                    attribute=field_name,
                    var=ScalarVariable(
                        name=config.pattern.format(
                            type=type_token, name=name_token, attrib=attrib_token
                        ),
                        default_value=impact.ele[ele_name][field_name],
                        unit=ELE_UNITS.get(field_name),
                        read_only=attr_cfg.read_only,
                    ),
                )
            )
    return actions


def _make_stat_actions(
    impact: Any, config: StatsConfig, stat_size_expansion: float
) -> list[Action]:
    actions = []
    for field_name, enabled in config.model_dump().items():
        if not isinstance(enabled, bool) or not enabled:
            continue
        name_token = config.attrib_map.get(field_name, field_name)
        stat_array = impact.stat(field_name)
        if config.max_size is not None:
            shape = (config.max_size,)
        else:
            shape = (int(stat_array.shape[0] * stat_size_expansion),)
        default_value = np.full(shape[0], np.nan, dtype=float)
        n = min(stat_array.shape[0], shape[0])
        default_value[:n] = stat_array[:n]
        actions.append(
            StatAction(
                stat_name=field_name,
                var=NDVariable(
                    name=config.pattern.format(name=name_token),
                    shape=shape,
                    default_value=default_value,
                    unit=STAT_UNITS.get(field_name),
                    read_only=True,
                ),
            )
        )
    return actions


def _make_run_info_actions(impact: Any, config: RunInfoConfig) -> list[Action]:
    actions = []
    run_info_data = impact.output.get("run_info", {})
    for field_name, enabled in config.model_dump().items():
        if not isinstance(enabled, bool) or not enabled:
            continue
        key_token = config.attrib_map.get(field_name, field_name)
        actions.append(
            RunInfoAction(
                key=field_name,
                var=ScalarVariable(
                    name=config.pattern.format(key=key_token),
                    default_value=run_info_data.get(field_name),
                    unit=RUN_INFO_UNITS.get(field_name),
                    read_only=True,
                ),
            )
        )
    return actions


def _make_particle_actions(impact: Any, config: ParticlesConfig) -> list[Action]:
    actions = []
    reverse_particle_map = {v: k for k, v in config.name_map.items()}
    particles_data = impact.output.get("particles", {})
    for tool_name in impact.particles.keys():
        control_name = reverse_particle_map.get(tool_name, tool_name)
        default_val = (
            getattr(impact, "initial_particles", None)
            if tool_name == "initial_particles"
            else particles_data.get(tool_name)
        )
        actions.append(
            ParticleGroupAction(
                tool_name=tool_name,
                var=ParticleGroupVariable(
                    name=config.pattern.format(name=control_name),
                    default_value=default_val,
                    read_only=tool_name != "initial_particles",
                ),
            )
        )
    return actions


def make_actions(
    impact: Impact,
    config: VariableMappingConfig,
    stat_size_expansion: float = 1.1,
) -> list[Action]:
    """Build variable actions for every element attribute, header key, and output
    described by *config*.

    Parameters
    ----------
    impact : Impact
        Already run Impact-T simulator object from which variables will be generated.
    config : VariableMappingConfig
        Configuration for how variables are generated from Impact-T
    stat_size_expansion : float
        Multiplier applied to the current stat array length to pre-size the
        NDVariable shape when ``StatsConfig.max_size`` is not set.
    """
    actions: list[Action] = []
    if config.header is not None:
        actions += _make_header_actions(impact, config.header)
    if config.elements is not None:
        actions += _make_element_actions(impact, config.elements)
    if config.stats is not None:
        actions += _make_stat_actions(impact, config.stats, stat_size_expansion)
    if config.run_info is not None:
        actions += _make_run_info_actions(impact, config.run_info)
    if config.particles is not None:
        actions += _make_particle_actions(impact, config.particles)
    return actions
