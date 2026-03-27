import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable
from impact.model.impact_actions import (
    ImpactVarAction,
    EleVarAction,
    HeaderVarAction,
    StatVarAction,
    RunInfoVarAction,
    ParticleGroupVarAction,
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


# ------------------------------------------------------------------
# Per-element-type configs
# Set a field to AttributeConfig() to include it; leave as None to exclude.
# ------------------------------------------------------------------

_DRIFT_DEFAULTS: dict[str, dict] = {
    "zedge": {"unit": "m"},
    "radius": {"unit": "m"},
}


class DriftConfig(BaseModel):
    zedge: AttributeConfig | None = AttributeConfig(**_DRIFT_DEFAULTS.get("zedge", {}))
    radius: AttributeConfig | None = AttributeConfig(
        **_DRIFT_DEFAULTS.get("radius", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _DRIFT_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_QUADRUPOLE_DEFAULTS: dict[str, dict] = {
    "b1_gradient": {"unit": "T/m"},
    "L_effective": {"unit": "m"},
    "radius": {"unit": "m"},
    "rf_frequency": {"unit": "Hz"},
    "rf_phase_deg": {"unit": "deg", "alias": "rf_phase"},
    "x_offset": {"unit": "m"},
    "y_offset": {"unit": "m"},
    "x_rotation": {"unit": "deg"},
    "y_rotation": {"unit": "deg"},
    "z_rotation": {"unit": "deg"},
}


class QuadrupoleConfig(BaseModel):
    b1_gradient: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("b1_gradient", {})
    )
    L_effective: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("L_effective", {})
    )
    radius: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("radius", {})
    )
    rf_frequency: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("rf_frequency", {})
    )
    rf_phase_deg: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("rf_phase_deg", {})
    )
    x_offset: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("x_offset", {})
    )
    y_offset: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("y_offset", {})
    )
    x_rotation: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("x_rotation", {})
    )
    y_rotation: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("y_rotation", {})
    )
    z_rotation: AttributeConfig | None = AttributeConfig(
        **_QUADRUPOLE_DEFAULTS.get("z_rotation", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _QUADRUPOLE_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_SOLENOID_DEFAULTS: dict[str, dict] = {
    "b_field": {"unit": "T"},
    "radius": {"unit": "m"},
    "x_offset": {"unit": "m"},
    "y_offset": {"unit": "m"},
    "x_rotation": {"unit": "deg"},
    "y_rotation": {"unit": "deg"},
    "z_rotation": {"unit": "deg"},
}


class SolenoidConfig(BaseModel):
    b_field: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("b_field", {})
    )
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("radius", {})
    )
    x_offset: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("x_offset", {})
    )
    y_offset: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("y_offset", {})
    )
    x_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("x_rotation", {})
    )
    y_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("y_rotation", {})
    )
    z_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLENOID_DEFAULTS.get("z_rotation", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _SOLENOID_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_DIPOLE_DEFAULTS: dict[str, dict] = {
    "b_field": {"unit": "T"},
    "b_field_x": {"unit": "T"},
    "half_gap": {"unit": "m"},
}


class DipoleConfig(BaseModel):
    b_field: AttributeConfig | None = AttributeConfig(
        **_DIPOLE_DEFAULTS.get("b_field", {})
    )
    b_field_x: AttributeConfig | None = AttributeConfig(
        **_DIPOLE_DEFAULTS.get("b_field_x", {})
    )
    #     filename: AttributeConfig | None = AttributeConfig()
    half_gap: AttributeConfig | None = AttributeConfig(
        **_DIPOLE_DEFAULTS.get("half_gap", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _DIPOLE_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_SOLRF_DEFAULTS: dict[str, dict] = {
    "rf_frequency": {"unit": "Hz"},
    "theta0_deg": {"unit": "deg", "alias": "theta0"},
    "radius": {"unit": "m"},
    "x_offset": {"unit": "m"},
    "y_offset": {"unit": "m"},
    "x_rotation": {"unit": "deg"},
    "y_rotation": {"unit": "deg"},
    "z_rotation": {"unit": "deg"},
}


class SolrfConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("rf_field_scale", {})
    )
    rf_frequency: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("rf_frequency", {})
    )
    theta0_deg: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("theta0_deg", {})
    )
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("radius", {})
    )
    solenoid_field_scale: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("solenoid_field_scale", {})
    )
    x_offset: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("x_offset", {})
    )
    y_offset: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("y_offset", {})
    )
    x_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("x_rotation", {})
    )
    y_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("y_rotation", {})
    )
    z_rotation: AttributeConfig | None = AttributeConfig(
        **_SOLRF_DEFAULTS.get("z_rotation", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _SOLRF_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_EMFIELD_CARTESIAN_DEFAULTS: dict[str, dict] = {
    "rf_frequency": {"unit": "Hz"},
    "theta0_deg": {"unit": "deg", "alias": "theta0"},
    "radius": {"unit": "m"},
    "x_offset": {"unit": "m"},
    "y_offset": {"unit": "m"},
    "x_rotation": {"unit": "deg"},
    "y_rotation": {"unit": "deg"},
    "z_rotation": {"unit": "deg"},
}


class EmfieldCartesianConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("rf_field_scale", {})
    )
    rf_frequency: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("rf_frequency", {})
    )
    theta0_deg: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("theta0_deg", {})
    )
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("radius", {})
    )
    x_offset: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("x_offset", {})
    )
    y_offset: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("y_offset", {})
    )
    x_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("x_rotation", {})
    )
    y_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("y_rotation", {})
    )
    z_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CARTESIAN_DEFAULTS.get("z_rotation", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _EMFIELD_CARTESIAN_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


_EMFIELD_CYLINDRICAL_DEFAULTS: dict[str, dict] = {
    "rf_frequency": {"unit": "Hz"},
    "theta0_deg": {"unit": "deg", "alias": "theta0"},
    "radius": {"unit": "m"},
    "x_offset": {"unit": "m"},
    "y_offset": {"unit": "m"},
    "x_rotation": {"unit": "deg"},
    "y_rotation": {"unit": "deg"},
    "z_rotation": {"unit": "deg"},
}


class EmfieldCylindricalConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("rf_field_scale", {})
    )
    rf_frequency: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("rf_frequency", {})
    )
    theta0_deg: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("theta0_deg", {})
    )
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("radius", {})
    )
    x_offset: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("x_offset", {})
    )
    y_offset: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("y_offset", {})
    )
    x_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("x_rotation", {})
    )
    y_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("y_rotation", {})
    )
    z_rotation: AttributeConfig | None = AttributeConfig(
        **_EMFIELD_CYLINDRICAL_DEFAULTS.get("z_rotation", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _EMFIELD_CYLINDRICAL_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


# ------------------------------------------------------------------
# Header config
# ------------------------------------------------------------------

_HEADER_DEFAULTS: dict[str, dict] = {
    # Time stepping
    "Dt": {"unit": "s"},
    # Beam / bunch
    "Bcurr": {"unit": "A"},
    "Bkenergy": {"unit": "eV"},
    "Bmass": {"unit": "eV"},
    "Bfreq": {"unit": "Hz"},
    "Tini": {"unit": "s"},
    # Grid
    "Xrad": {"unit": "m"},
    "Yrad": {"unit": "m"},
    "Perdlen": {"unit": "m"},
    "Zimage": {"unit": "m"},
    # Emission
    "Temission": {"unit": "s"},
    # Distribution parameters — alias strips the "(m)" from the header key
    "sigx_m": {"unit": "m", "alias": "sigx"},
    "xmu1_m": {"unit": "m", "alias": "xmu1"},
    "sigy_m": {"unit": "m", "alias": "sigy"},
    "ymu1_m": {"unit": "m", "alias": "ymu1"},
    "sigz_m": {"unit": "m", "alias": "sigz"},
    "zmu1_m": {"unit": "m", "alias": "zmu1"},
}


class HeaderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pattern: str = "header/{key}"
    regex: str | None = None

    @property
    def key_map(self) -> dict[str, str]:
        """Maps variable token (alias) -> actual imp.header key, for aliased fields."""
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
    Npcol: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Npcol", {}))
    Nprow: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Nprow", {}))
    # Time stepping
    Dt: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Dt", {}))
    Ntstep: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Ntstep", {})
    )
    # Beam / bunch
    Nbunch: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Nbunch", {})
    )
    Np: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Np", {}))
    Bcurr: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Bcurr", {}))
    Bkenergy: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Bkenergy", {})
    )
    Bmass: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Bmass", {}))
    Bcharge: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Bcharge", {})
    )
    Bfreq: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Bfreq", {}))
    Tini: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Tini", {}))
    # Flags
    Flagmap: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagmap", {})
    )
    Flagerr: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagerr", {})
    )
    Flagdiag: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagdiag", {})
    )
    Flagimg: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagimg", {})
    )
    Flagdist: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagdist", {})
    )
    Flagbc: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagbc", {})
    )
    Rstartflg: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Rstartflg", {})
    )
    Flagsbstp: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Flagsbstp", {})
    )
    # Grid
    Dim: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Dim", {}))
    Nx: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Nx", {}))
    Ny: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Ny", {}))
    Nz: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Nz", {}))
    Xrad: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Xrad", {}))
    Yrad: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("Yrad", {}))
    Perdlen: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Perdlen", {})
    )
    Zimage: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Zimage", {})
    )
    # Emission
    Nemission: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Nemission", {})
    )
    Temission: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("Temission", {})
    )
    # Distribution parameters
    sigx_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("sigx_m", {})), alias="sigx(m)"
    )
    sigpx: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("sigpx", {}))
    muxpx: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("muxpx", {}))
    xscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("xscale", {})
    )
    pxscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("pxscale", {})
    )
    xmu1_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("xmu1_m", {})), alias="xmu1(m)"
    )
    xmu2: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("xmu2", {}))
    sigy_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("sigy_m", {})), alias="sigy(m)"
    )
    sigpy: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("sigpy", {}))
    muxpy: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("muxpy", {}))
    yscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("yscale", {})
    )
    pyscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("pyscale", {})
    )
    ymu1_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("ymu1_m", {})), alias="ymu1(m)"
    )
    ymu2: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("ymu2", {}))
    sigz_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("sigz_m", {})), alias="sigz(m)"
    )
    sigpz: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("sigpz", {}))
    muxpz: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("muxpz", {}))
    zscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("zscale", {})
    )
    pzscale: AttributeConfig | None = AttributeConfig(
        **_HEADER_DEFAULTS.get("pzscale", {})
    )
    zmu1_m: AttributeConfig | None = Field(
        AttributeConfig(**_HEADER_DEFAULTS.get("zmu1_m", {})), alias="zmu1(m)"
    )
    zmu2: AttributeConfig | None = AttributeConfig(**_HEADER_DEFAULTS.get("zmu2", {}))

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _HEADER_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


# ------------------------------------------------------------------
# Stats config
# ------------------------------------------------------------------

_STATS_DEFAULTS: dict[str, dict] = {
    "mean_kinetic_energy": {"unit": "eV"},
    "mean_x": {"unit": "m"},
    "mean_y": {"unit": "m"},
    "mean_z": {"unit": "m"},
    "sigma_x": {"unit": "m"},
    "sigma_y": {"unit": "m"},
    "sigma_z": {"unit": "m"},
    "norm_emit_x": {"unit": "m"},
    "norm_emit_y": {"unit": "m"},
    "norm_emit_z": {"unit": "m"},
}


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


class StatsConfig(BaseModel):
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

    mean_kinetic_energy: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("mean_kinetic_energy", {})
    )
    mean_x: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("mean_x", {})
    )
    mean_y: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("mean_y", {})
    )
    mean_z: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("mean_z", {})
    )
    sigma_x: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("sigma_x", {})
    )
    sigma_y: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("sigma_y", {})
    )
    sigma_z: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("sigma_z", {})
    )
    norm_emit_x: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("norm_emit_x", {})
    )
    norm_emit_y: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("norm_emit_y", {})
    )
    norm_emit_z: StatAttributeConfig | None = StatAttributeConfig(
        **_STATS_DEFAULTS.get("norm_emit_z", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _STATS_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


# ------------------------------------------------------------------
# Run info config
# ------------------------------------------------------------------

_RUN_INFO_DEFAULTS: dict[str, dict] = {
    "run_time": {"unit": "s"},
    "error": {},
}


class RunInfoConfig(BaseModel):
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

    run_time: StatAttributeConfig | None = StatAttributeConfig(
        **_RUN_INFO_DEFAULTS.get("run_time", {})
    )
    error: StatAttributeConfig | None = StatAttributeConfig(
        **_RUN_INFO_DEFAULTS.get("error", {})
    )

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field, default in _RUN_INFO_DEFAULTS.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


# ------------------------------------------------------------------
# Element container + top-level mapping config
# ------------------------------------------------------------------


class ElementsConfig(BaseModel):
    """Groups all per-element-type attribute configs and element routing settings.

    Set a type to ``None`` to skip all variables for that element type.
    """

    pattern: str = "ele/{name}/{attrib}"
    regex: str | None = None
    name_mappings: dict[str, str] | None = None  # control_name -> tool_name
    type_mappings: dict[str, str] | None = None  # control_type -> tool_type

    drift: DriftConfig | None = DriftConfig()
    quadrupole: QuadrupoleConfig | None = QuadrupoleConfig()
    solenoid: SolenoidConfig | None = SolenoidConfig()
    dipole: DipoleConfig | None = DipoleConfig()
    solrf: SolrfConfig | None = SolrfConfig()
    emfield_cartesian: EmfieldCartesianConfig | None = EmfieldCartesianConfig()
    emfield_cylindrical: EmfieldCylindricalConfig | None = EmfieldCylindricalConfig()

    @property
    def attrib_map(self) -> dict[str, str]:
        """Maps attrib token (alias) -> actual imp.ele field name, for aliased attributes."""
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
    name_mappings: dict[str, str] | None = None  # control_name -> tool_name

    @property
    def name_map(self) -> dict[str, str]:
        """Maps particle variable token (control name) -> actual key in imp.particles."""
        return self.name_mappings or {}


class VariableMappingConfig(BaseModel):
    """Maps Impact-T element attributes, header keys, and output stats to model variable names.

    Set any sub-config to ``None`` to skip that category entirely.
    """

    header: HeaderConfig | None = HeaderConfig()
    elements: ElementsConfig | None = ElementsConfig()
    stats: StatsConfig | None = StatsConfig()
    run_info: RunInfoConfig | None = RunInfoConfig()
    particles: ParticlesConfig | None = ParticlesConfig()


def make_variables(imp: Any, config: VariableMappingConfig) -> list[ImpactVarAction]:
    """Build variable mappings for every element attribute, header key, and output
    described by *config*.

    The current value in *imp* is used as ``default_value`` for each variable.
    """
    mappings: list[ImpactVarAction] = []

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
                HeaderVarAction(
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
                    EleVarAction(
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
                StatVarAction(
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
                RunInfoVarAction(
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
                ParticleGroupVarAction(
                    tool_name=tool_name,
                    var=ParticleGroupVariable(
                        name=variable_name,
                        default_value=default_val,
                        read_only=tool_name != "initial_particles",
                    ),
                )
            )

    return mappings
