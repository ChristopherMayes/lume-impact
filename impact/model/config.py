import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lume.variables import ScalarVariable


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
    """

    alias: str | None = None
    unit: str | None = None


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
# Top-level mapping config
# ------------------------------------------------------------------


class VariableMappingConfig(BaseModel):
    """Maps Impact-T element attributes and header keys to model variable names.

    Parameters
    ----------
    header_pattern : str
        Format string for header variable names. Available token: ``{key}``.
        Example: ``"header_{key}"`` -> ``"header_Np"``.
    element_pattern : str
        Format string for element variable names.
        Available tokens: ``{type}``, ``{name}``, ``{attrib}``.
        Example: ``"{name}_{attrib}"`` -> ``"Q1_b1_gradient"``.

    header :
        Header key mappings. ``None`` skips all header variables.
    drift, quadrupole, solenoid, dipole, solrf, emfield_cartesian, emfield_cylindrical :
        Per-type config. ``None`` skips that element type entirely.
        Within each type, attributes left as ``None`` are not registered.
    """

    header_pattern: str = "header_{key}"
    element_pattern: str = "ele_{name}_{attrib}"

    header: HeaderConfig | None = HeaderConfig()
    drift: DriftConfig | None = DriftConfig()
    quadrupole: QuadrupoleConfig | None = QuadrupoleConfig()
    solenoid: SolenoidConfig | None = SolenoidConfig()
    dipole: DipoleConfig | None = DipoleConfig()
    solrf: SolrfConfig | None = SolrfConfig()
    emfield_cartesian: EmfieldCartesianConfig | None = EmfieldCartesianConfig()
    emfield_cylindrical: EmfieldCylindricalConfig | None = EmfieldCylindricalConfig()


def make_variables(imp: Any, config: VariableMappingConfig) -> list[ScalarVariable]:
    """Build a ``ScalarVariable`` for every element attribute and header key
    described by *config*. The current value in *imp* is used as ``default_value``.
    """
    variables = []

    if config.header is not None:
        for field_name, field_info in HeaderConfig.model_fields.items():
            attr_cfg: AttributeConfig | None = getattr(config.header, field_name)
            if attr_cfg is None:
                continue

            # Field alias is the actual Impact-T header key (e.g. "sigx(m)")
            header_key = (
                field_info.alias if field_info.alias is not None else field_name
            )
            key_token = attr_cfg.alias if attr_cfg.alias is not None else header_key
            variable_name = config.header_pattern.format(key=key_token)

            variables.append(
                ScalarVariable(
                    name=variable_name,
                    default_value=imp.header.get(header_key),
                    unit=attr_cfg.unit,
                    read_only=False,
                )
            )

    for ele in imp.lattice:
        ele_type: str = ele.get("type", "")
        ele_name: str = ele.get("name", "")

        type_cfg = getattr(config, ele_type, None)
        if type_cfg is None:
            continue

        for field_name in type_cfg.model_fields:
            attr_cfg: AttributeConfig | None = getattr(type_cfg, field_name)
            if attr_cfg is None:
                continue

            if field_name not in imp.ele[ele_name]:
                continue

            attrib_token = attr_cfg.alias if attr_cfg.alias is not None else field_name
            variable_name = config.element_pattern.format(
                type=ele_type, name=ele_name, attrib=attrib_token
            )

            variables.append(
                ScalarVariable(
                    name=variable_name,
                    default_value=imp.ele[ele_name][field_name],
                    unit=attr_cfg.unit,
                    read_only=False,
                )
            )

    return variables
