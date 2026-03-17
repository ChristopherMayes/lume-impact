from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lume.variables import ScalarVariable


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


class QuadrupoleConfig(BaseModel):
    b1_gradient: AttributeConfig | None = None
    L_effective: AttributeConfig | None = None
    radius: AttributeConfig | None = None
    rf_frequency: AttributeConfig | None = None
    rf_phase_deg: AttributeConfig | None = None
    x_offset: AttributeConfig | None = None
    y_offset: AttributeConfig | None = None
    x_rotation: AttributeConfig | None = None
    y_rotation: AttributeConfig | None = None
    z_rotation: AttributeConfig | None = None


class SolenoidConfig(BaseModel):
    b_field: AttributeConfig | None = None
    filename: AttributeConfig | None = None
    radius: AttributeConfig | None = None
    x_offset: AttributeConfig | None = None
    y_offset: AttributeConfig | None = None
    x_rotation: AttributeConfig | None = None
    y_rotation: AttributeConfig | None = None
    z_rotation: AttributeConfig | None = None


class DipoleConfig(BaseModel):
    b_field: AttributeConfig | None = None
    b_field_x: AttributeConfig | None = None
    filename: AttributeConfig | None = None
    half_gap: AttributeConfig | None = None


class SolrfConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = None
    rf_frequency: AttributeConfig | None = None
    theta0_deg: AttributeConfig | None = None
    filename: AttributeConfig | None = None
    radius: AttributeConfig | None = None
    solenoid_field_scale: AttributeConfig | None = None
    x_offset: AttributeConfig | None = None
    y_offset: AttributeConfig | None = None
    x_rotation: AttributeConfig | None = None
    y_rotation: AttributeConfig | None = None
    z_rotation: AttributeConfig | None = None


class EmfieldCartesianConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = None
    rf_frequency: AttributeConfig | None = None
    theta0_deg: AttributeConfig | None = None
    filename: AttributeConfig | None = None
    radius: AttributeConfig | None = None
    x_offset: AttributeConfig | None = None
    y_offset: AttributeConfig | None = None
    x_rotation: AttributeConfig | None = None
    y_rotation: AttributeConfig | None = None
    z_rotation: AttributeConfig | None = None


class EmfieldCylindricalConfig(BaseModel):
    rf_field_scale: AttributeConfig | None = None
    rf_frequency: AttributeConfig | None = None
    theta0_deg: AttributeConfig | None = None
    filename: AttributeConfig | None = None
    radius: AttributeConfig | None = None
    x_offset: AttributeConfig | None = None
    y_offset: AttributeConfig | None = None
    x_rotation: AttributeConfig | None = None
    y_rotation: AttributeConfig | None = None
    z_rotation: AttributeConfig | None = None


# ------------------------------------------------------------------
# Header config
# ------------------------------------------------------------------


class HeaderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Processor domain
    Npcol: AttributeConfig | None = None
    Nprow: AttributeConfig | None = None
    # Time stepping
    Dt: AttributeConfig | None = None
    Ntstep: AttributeConfig | None = None
    # Beam / bunch
    Nbunch: AttributeConfig | None = None
    Np: AttributeConfig | None = None
    Bcurr: AttributeConfig | None = None
    Bkenergy: AttributeConfig | None = None
    Bmass: AttributeConfig | None = None
    Bcharge: AttributeConfig | None = None
    Bfreq: AttributeConfig | None = None
    Tini: AttributeConfig | None = None
    # Flags
    Flagmap: AttributeConfig | None = None
    Flagerr: AttributeConfig | None = None
    Flagdiag: AttributeConfig | None = None
    Flagimg: AttributeConfig | None = None
    Flagdist: AttributeConfig | None = None
    Flagbc: AttributeConfig | None = None
    Rstartflg: AttributeConfig | None = None
    Flagsbstp: AttributeConfig | None = None
    # Grid
    Dim: AttributeConfig | None = None
    Nx: AttributeConfig | None = None
    Ny: AttributeConfig | None = None
    Nz: AttributeConfig | None = None
    Xrad: AttributeConfig | None = None
    Yrad: AttributeConfig | None = None
    Perdlen: AttributeConfig | None = None
    Zimage: AttributeConfig | None = None
    # Emission
    Nemission: AttributeConfig | None = None
    Temission: AttributeConfig | None = None
    # Distribution parameters
    sigx_m: AttributeConfig | None = Field(None, alias="sigx(m)")
    sigpx: AttributeConfig | None = None
    muxpx: AttributeConfig | None = None
    xscale: AttributeConfig | None = None
    pxscale: AttributeConfig | None = None
    xmu1_m: AttributeConfig | None = Field(None, alias="xmu1(m)")
    xmu2: AttributeConfig | None = None
    sigy_m: AttributeConfig | None = Field(None, alias="sigy(m)")
    sigpy: AttributeConfig | None = None
    muxpy: AttributeConfig | None = None
    yscale: AttributeConfig | None = None
    pyscale: AttributeConfig | None = None
    ymu1_m: AttributeConfig | None = Field(None, alias="ymu1(m)")
    ymu2: AttributeConfig | None = None
    sigz_m: AttributeConfig | None = Field(None, alias="sigz(m)")
    sigpz: AttributeConfig | None = None
    muxpz: AttributeConfig | None = None
    zscale: AttributeConfig | None = None
    pzscale: AttributeConfig | None = None
    zmu1_m: AttributeConfig | None = Field(None, alias="zmu1(m)")
    zmu2: AttributeConfig | None = None


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

    header: HeaderConfig | None = None
    drift: DriftConfig | None = None
    quadrupole: QuadrupoleConfig | None = None
    solenoid: SolenoidConfig | None = None
    dipole: DipoleConfig | None = None
    solrf: SolrfConfig | None = None
    emfield_cartesian: EmfieldCartesianConfig | None = None
    emfield_cylindrical: EmfieldCylindricalConfig | None = None


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
