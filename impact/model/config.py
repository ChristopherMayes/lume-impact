from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable
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
    """Config for a single element attribute.

    Parameters
    ----------
    alias : str, optional
        Name to substitute for ``{attrib}`` in the pattern.
        If omitted, the attribute name itself is used.
    unit : str, optional
        Physical unit string passed to ``ScalarVariable``.
    read_only : bool, optional
        Whether the variable is read-only.  Defaults to ``False``.
    """

    alias: str | None = None
    unit: str | None = None
    read_only: bool = False


class StatAttributeConfig(BaseModel):
    """Config for a single output stat variable.

    Parameters
    ----------
    unit : str, optional
        Physical unit string passed to ``ScalarVariable``.
    alias : str, optional
        Override the token used in the variable name pattern. If omitted the
        field name is used (e.g. ``"mean_x"``).
    """

    unit: str | None = None
    alias: str | None = None


class ConfigBase(BaseModel):
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
    zedge: AttributeConfig | None = AttributeConfig(unit="m")
    radius: AttributeConfig | None = AttributeConfig(unit="m")


class QuadrupoleConfig(ConfigBase):
    b1_gradient: AttributeConfig | None = AttributeConfig(unit="T/m")
    L_effective: AttributeConfig | None = AttributeConfig(unit="m")
    radius: AttributeConfig | None = AttributeConfig(unit="m")
    rf_frequency: AttributeConfig | None = AttributeConfig(unit="Hz")
    rf_phase_deg: AttributeConfig | None = AttributeConfig(unit="deg", alias="rf_phase")
    x_offset: AttributeConfig | None = AttributeConfig(unit="m")
    y_offset: AttributeConfig | None = AttributeConfig(unit="m")
    x_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    y_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    z_rotation: AttributeConfig | None = AttributeConfig(unit="deg")


class SolenoidConfig(ConfigBase):
    b_field: AttributeConfig | None = AttributeConfig(unit="T")
    radius: AttributeConfig | None = AttributeConfig(unit="m")
    x_offset: AttributeConfig | None = AttributeConfig(unit="m")
    y_offset: AttributeConfig | None = AttributeConfig(unit="m")
    x_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    y_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    z_rotation: AttributeConfig | None = AttributeConfig(unit="deg")


class DipoleConfig(ConfigBase):
    b_field: AttributeConfig | None = AttributeConfig(unit="T")
    b_field_x: AttributeConfig | None = AttributeConfig(unit="T")
    half_gap: AttributeConfig | None = AttributeConfig(unit="m")


class SolrfConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig(unit="Hz")
    theta0_deg: AttributeConfig | None = AttributeConfig(unit="deg", alias="theta0")
    radius: AttributeConfig | None = AttributeConfig(unit="m")
    solenoid_field_scale: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig(unit="m")
    y_offset: AttributeConfig | None = AttributeConfig(unit="m")
    x_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    y_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    z_rotation: AttributeConfig | None = AttributeConfig(unit="deg")


class EmfieldCartesianConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig(unit="Hz")
    theta0_deg: AttributeConfig | None = AttributeConfig(unit="deg", alias="theta0")
    radius: AttributeConfig | None = AttributeConfig(unit="m")
    x_offset: AttributeConfig | None = AttributeConfig(unit="m")
    y_offset: AttributeConfig | None = AttributeConfig(unit="m")
    x_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    y_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    z_rotation: AttributeConfig | None = AttributeConfig(unit="deg")


class EmfieldCylindricalConfig(ConfigBase):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig(unit="Hz")
    theta0_deg: AttributeConfig | None = AttributeConfig(unit="deg", alias="theta0")
    radius: AttributeConfig | None = AttributeConfig(unit="m")
    x_offset: AttributeConfig | None = AttributeConfig(unit="m")
    y_offset: AttributeConfig | None = AttributeConfig(unit="m")
    x_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    y_rotation: AttributeConfig | None = AttributeConfig(unit="deg")
    z_rotation: AttributeConfig | None = AttributeConfig(unit="deg")


# ------------------------------------------------------------------
# Header config
# ------------------------------------------------------------------


class HeaderConfig(ConfigBase):
    model_config = ConfigDict(populate_by_name=True)

    pattern: str = "header/{key}"

    @property
    def key_map(self) -> dict[str, str]:
        """Maps variable token (alias) -> actual impact.header key, for aliased fields."""
        result = {}
        for field_name, field_info in HeaderConfig.model_fields.items():
            attr_cfg = getattr(self, field_name)
            if not isinstance(attr_cfg, AttributeConfig):
                continue
            header_key = (
                field_info.alias if field_info.alias is not None else field_name
            )
            key_token = attr_cfg.alias if attr_cfg.alias is not None else header_key
            if key_token != header_key:
                result[key_token] = header_key
        return result

    # Processor domain
    Npcol: AttributeConfig | None = AttributeConfig()
    Nprow: AttributeConfig | None = AttributeConfig()
    # Time stepping
    Dt: AttributeConfig | None = AttributeConfig(unit="s")
    Ntstep: AttributeConfig | None = AttributeConfig()
    # Beam / bunch
    Nbunch: AttributeConfig | None = AttributeConfig()
    Np: AttributeConfig | None = AttributeConfig()
    Bcurr: AttributeConfig | None = AttributeConfig(unit="A")
    Bkenergy: AttributeConfig | None = AttributeConfig(unit="eV")
    Bmass: AttributeConfig | None = AttributeConfig(unit="eV")
    Bcharge: AttributeConfig | None = AttributeConfig()
    Bfreq: AttributeConfig | None = AttributeConfig(unit="Hz")
    Tini: AttributeConfig | None = AttributeConfig(unit="s")
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
    Xrad: AttributeConfig | None = AttributeConfig(unit="m")
    Yrad: AttributeConfig | None = AttributeConfig(unit="m")
    Perdlen: AttributeConfig | None = AttributeConfig(unit="m")
    Zimage: AttributeConfig | None = AttributeConfig(unit="m")
    # Emission
    Nemission: AttributeConfig | None = AttributeConfig()
    Temission: AttributeConfig | None = AttributeConfig(unit="s")
    # Distribution parameters
    sigx_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="sigx"), alias="sigx(m)"
    )
    sigpx: AttributeConfig | None = AttributeConfig()
    muxpx: AttributeConfig | None = AttributeConfig()
    xscale: AttributeConfig | None = AttributeConfig()
    pxscale: AttributeConfig | None = AttributeConfig()
    xmu1_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="xmu1"), alias="xmu1(m)"
    )
    xmu2: AttributeConfig | None = AttributeConfig()
    sigy_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="sigy"), alias="sigy(m)"
    )
    sigpy: AttributeConfig | None = AttributeConfig()
    muxpy: AttributeConfig | None = AttributeConfig()
    yscale: AttributeConfig | None = AttributeConfig()
    pyscale: AttributeConfig | None = AttributeConfig()
    ymu1_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="ymu1"), alias="ymu1(m)"
    )
    ymu2: AttributeConfig | None = AttributeConfig()
    sigz_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="sigz"), alias="sigz(m)"
    )
    sigpz: AttributeConfig | None = AttributeConfig()
    muxpz: AttributeConfig | None = AttributeConfig()
    zscale: AttributeConfig | None = AttributeConfig()
    pzscale: AttributeConfig | None = AttributeConfig()
    zmu1_m: AttributeConfig | None = Field(
        AttributeConfig(unit="m", alias="zmu1"), alias="zmu1(m)"
    )
    zmu2: AttributeConfig | None = AttributeConfig()


# ------------------------------------------------------------------
# Stats config
# ------------------------------------------------------------------


class StatsConfig(ConfigBase):
    pattern: str = "stat/{name}"

    @property
    def name_map(self) -> dict[str, str]:
        """Maps variable token (alias) -> actual Impact stat key, for aliased stats."""
        result = {}
        for field_name in StatsConfig.model_fields:
            stat_cfg = getattr(self, field_name)
            if not isinstance(stat_cfg, StatAttributeConfig):
                continue
            if stat_cfg.alias is not None and stat_cfg.alias != field_name:
                result[stat_cfg.alias] = field_name
        return result

    mean_kinetic_energy: StatAttributeConfig | None = StatAttributeConfig(unit="eV")
    mean_x: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    mean_y: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    mean_z: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    sigma_x: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    sigma_y: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    sigma_z: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    norm_emit_x: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    norm_emit_y: StatAttributeConfig | None = StatAttributeConfig(unit="m")
    norm_emit_z: StatAttributeConfig | None = StatAttributeConfig(unit="m")


# ------------------------------------------------------------------
# Run info config
# ------------------------------------------------------------------


class RunInfoConfig(ConfigBase):
    pattern: str = "run_info/{key}"

    @property
    def key_map(self) -> dict[str, str]:
        """Maps variable token (alias) -> actual run_info key, for aliased fields."""
        result = {}
        for field_name in RunInfoConfig.model_fields:
            stat_cfg = getattr(self, field_name)
            if not isinstance(stat_cfg, StatAttributeConfig):
                continue
            if stat_cfg.alias is not None and stat_cfg.alias != field_name:
                result[stat_cfg.alias] = field_name
        return result

    run_time: StatAttributeConfig | None = StatAttributeConfig(unit="s")
    error: StatAttributeConfig | None = StatAttributeConfig()


# ------------------------------------------------------------------
# Element container + top-level mapping config
# ------------------------------------------------------------------


class ElementsConfig(BaseModel):
    """Groups all per-element-type attribute configs and element routing settings.

    Set a type to ``None`` to skip all variables for that element type.
    """

    pattern: str = "ele/{name}/{attrib}"
    control_to_tool_name: dict[str, str] | None = None  # control_name -> tool_name
    control_to_tool_type: dict[str, str] | None = None  # control_type -> tool_type

    drift: DriftConfig | None = DriftConfig()
    quadrupole: QuadrupoleConfig | None = QuadrupoleConfig()
    solenoid: SolenoidConfig | None = SolenoidConfig()
    dipole: DipoleConfig | None = DipoleConfig()
    solrf: SolrfConfig | None = SolrfConfig()
    emfield_cartesian: EmfieldCartesianConfig | None = EmfieldCartesianConfig()
    emfield_cylindrical: EmfieldCylindricalConfig | None = EmfieldCylindricalConfig()

    @property
    def attrib_map(self) -> dict[str, str]:
        """Maps attrib token (alias) -> actual impact.ele field name, for aliased attributes."""
        result = {}
        for type_field in ElementsConfig.model_fields:
            type_cfg = getattr(self, type_field)
            if not isinstance(type_cfg, BaseModel):
                continue
            for field_name in type(type_cfg).model_fields:
                attr_cfg = getattr(type_cfg, field_name)
                if not isinstance(attr_cfg, AttributeConfig):
                    continue
                if attr_cfg.alias is not None and attr_cfg.alias != field_name:
                    result[attr_cfg.alias] = field_name
        return result


class ParticlesConfig(BaseModel):
    """Config for particle group variables."""

    pattern: str = "particles/{name}"
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
        key_token = attr_cfg.alias if attr_cfg.alias is not None else header_key
        actions.append(
            HeaderAction(
                key=header_key,
                var=ScalarVariable(
                    name=config.pattern.format(key=key_token),
                    default_value=impact.header.get(header_key),
                    unit=attr_cfg.unit,
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
            attrib_token = attr_cfg.alias or field_name
            actions.append(
                EleAction(
                    ele_name=ele_name,
                    attribute=field_name,
                    var=ScalarVariable(
                        name=config.pattern.format(
                            type=type_token, name=name_token, attrib=attrib_token
                        ),
                        default_value=impact.ele[ele_name][field_name],
                        unit=attr_cfg.unit,
                        read_only=attr_cfg.read_only,
                    ),
                )
            )
    return actions


def _make_stat_actions(impact: Any, config: StatsConfig) -> list[Action]:
    actions = []
    for field_name in StatsConfig.model_fields:
        stat_cfg = getattr(config, field_name)
        if not isinstance(stat_cfg, StatAttributeConfig):
            continue
        name_token = stat_cfg.alias if stat_cfg.alias is not None else field_name
        stat_array = impact.stat(field_name)
        actions.append(
            StatAction(
                stat_name=field_name,
                var=NDVariable(
                    name=config.pattern.format(name=name_token),
                    shape=stat_array.shape,
                    default_value=stat_array,
                    unit=stat_cfg.unit,
                    read_only=True,
                ),
            )
        )
    return actions


def _make_run_info_actions(impact: Any, config: RunInfoConfig) -> list[Action]:
    actions = []
    run_info_data = impact.output.get("run_info", {})
    for field_name in RunInfoConfig.model_fields:
        run_info_cfg = getattr(config, field_name)
        if not isinstance(run_info_cfg, StatAttributeConfig):
            continue
        key_token = run_info_cfg.alias if run_info_cfg.alias is not None else field_name
        actions.append(
            RunInfoAction(
                key=field_name,
                var=ScalarVariable(
                    name=config.pattern.format(key=key_token),
                    default_value=run_info_data.get(field_name),
                    unit=run_info_cfg.unit,
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


def make_actions(impact: Any, config: VariableMappingConfig) -> list[Action]:
    """Build variable actions for every element attribute, header key, and output
    described by *config*.

    The current value in *impact* is used as ``default_value`` for each variable.
    """
    actions: list[Action] = []
    if config.header is not None:
        actions += _make_header_actions(impact, config.header)
    if config.elements is not None:
        actions += _make_element_actions(impact, config.elements)
    if config.stats is not None:
        actions += _make_stat_actions(impact, config.stats)
    if config.run_info is not None:
        actions += _make_run_info_actions(impact, config.run_info)
    if config.particles is not None:
        actions += _make_particle_actions(impact, config.particles)
    return actions
