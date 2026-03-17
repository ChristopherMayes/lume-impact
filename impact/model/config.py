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


class DriftConfig(BaseModel):
    zedge: AttributeConfig | None = None
    radius: AttributeConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "zedge": {"unit": "m"},
            "radius": {"unit": "m"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class QuadrupoleConfig(BaseModel):
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

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
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
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class SolenoidConfig(BaseModel):
    b_field: AttributeConfig | None = AttributeConfig()
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "b_field": {"unit": "T"},
            "radius": {"unit": "m"},
            "x_offset": {"unit": "m"},
            "y_offset": {"unit": "m"},
            "x_rotation": {"unit": "deg"},
            "y_rotation": {"unit": "deg"},
            "z_rotation": {"unit": "deg"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class DipoleConfig(BaseModel):
    b_field: AttributeConfig | None = AttributeConfig()
    b_field_x: AttributeConfig | None = AttributeConfig()
    #     filename: AttributeConfig | None = AttributeConfig()
    half_gap: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "b_field": {"unit": "T"},
            "b_field_x": {"unit": "T"},
            "half_gap": {"unit": "m"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class SolrfConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    solenoid_field_scale: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "rf_frequency": {"unit": "Hz"},
            "theta0_deg": {"unit": "deg", "alias": "theta0"},
            "radius": {"unit": "m"},
            "x_offset": {"unit": "m"},
            "y_offset": {"unit": "m"},
            "x_rotation": {"unit": "deg"},
            "y_rotation": {"unit": "deg"},
            "z_rotation": {"unit": "deg"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class EmfieldCartesianConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "rf_frequency": {"unit": "Hz"},
            "theta0_deg": {"unit": "deg", "alias": "theta0"},
            "radius": {"unit": "m"},
            "x_offset": {"unit": "m"},
            "y_offset": {"unit": "m"},
            "x_rotation": {"unit": "deg"},
            "y_rotation": {"unit": "deg"},
            "z_rotation": {"unit": "deg"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


class EmfieldCylindricalConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = AttributeConfig()
    rf_frequency: AttributeConfig | None = AttributeConfig()
    theta0_deg: AttributeConfig | None = AttributeConfig()
    #     filename: AttributeConfig | None = AttributeConfig()
    radius: AttributeConfig | None = AttributeConfig()
    x_offset: AttributeConfig | None = AttributeConfig()
    y_offset: AttributeConfig | None = AttributeConfig()
    x_rotation: AttributeConfig | None = AttributeConfig()
    y_rotation: AttributeConfig | None = AttributeConfig()
    z_rotation: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
            "rf_frequency": {"unit": "Hz"},
            "theta0_deg": {"unit": "deg", "alias": "theta0"},
            "radius": {"unit": "m"},
            "x_offset": {"unit": "m"},
            "y_offset": {"unit": "m"},
            "x_rotation": {"unit": "deg"},
            "y_rotation": {"unit": "deg"},
            "z_rotation": {"unit": "deg"},
        }
        for field, default in defaults.items():
            if field in data and isinstance(data[field], dict):
                data[field] = {**default, **data[field]}
        return data


# ------------------------------------------------------------------
# Header config
# ------------------------------------------------------------------


class HeaderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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
    sigx_m: AttributeConfig | None = Field(AttributeConfig(), alias="sigx(m)")
    sigpx: AttributeConfig | None = AttributeConfig()
    muxpx: AttributeConfig | None = AttributeConfig()
    xscale: AttributeConfig | None = AttributeConfig()
    pxscale: AttributeConfig | None = AttributeConfig()
    xmu1_m: AttributeConfig | None = Field(AttributeConfig(), alias="xmu1(m)")
    xmu2: AttributeConfig | None = AttributeConfig()
    sigy_m: AttributeConfig | None = Field(AttributeConfig(), alias="sigy(m)")
    sigpy: AttributeConfig | None = AttributeConfig()
    muxpy: AttributeConfig | None = AttributeConfig()
    yscale: AttributeConfig | None = AttributeConfig()
    pyscale: AttributeConfig | None = AttributeConfig()
    ymu1_m: AttributeConfig | None = Field(AttributeConfig(), alias="ymu1(m)")
    ymu2: AttributeConfig | None = AttributeConfig()
    sigz_m: AttributeConfig | None = Field(AttributeConfig(), alias="sigz(m)")
    sigpz: AttributeConfig | None = AttributeConfig()
    muxpz: AttributeConfig | None = AttributeConfig()
    zscale: AttributeConfig | None = AttributeConfig()
    pzscale: AttributeConfig | None = AttributeConfig()
    zmu1_m: AttributeConfig | None = Field(AttributeConfig(), alias="zmu1(m)")
    zmu2: AttributeConfig | None = AttributeConfig()

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        defaults = {
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
        for field, default in defaults.items():
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

            attrib_token = attr_cfg.alias if attr_cfg.alias is not None else field_name
            variable_name = config.element_pattern.format(
                type=ele_type, name=ele_name, attrib=attrib_token
            )

            variables.append(
                ScalarVariable(
                    name=variable_name,
                    default_value=imp.ele[ele_name].get(field_name),
                    unit=attr_cfg.unit,
                    read_only=False,
                )
            )

    return variables
