from pydantic import BaseModel


class AttributeConfig(BaseModel):
    """Config for a single element attribute.

    Parameters
    ----------
    alias : str, optional
        Name to substitute for ``{attrib}`` in the pattern.
        If omitted, the attribute name itself is used.
    """

    alias: str | None = None


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
# Top-level mapping config
# ------------------------------------------------------------------


class VariableMappingConfig(BaseModel):
    """Maps Impact-T element attributes to model variable names.

    Parameters
    ----------
    pattern : str
        Python format string for variable name generation.
        Available tokens: ``{type}``, ``{name}``, ``{attrib}``.

        Examples::

            "{name}_{attrib}"         ->  "Q1_b1_gradient"
            "{type}_{name}_{attrib}"  ->  "quadrupole_Q1_b1_gradient"
            "{name}:{attrib}"         ->  "Q1:b1_gradient"

    drift, quadrupole, solenoid, dipole, solrf, emfield_cartesian, emfield_cylindrical :
        Per-type config. ``None`` means that type is skipped entirely.
        Within each type, attributes left as ``None`` are not registered.
    """

    pattern: str = "{name}_{attrib}"

    drift: DriftConfig | None = None
    quadrupole: QuadrupoleConfig | None = None
    solenoid: SolenoidConfig | None = None
    dipole: DipoleConfig | None = None
    solrf: SolrfConfig | None = None
    emfield_cartesian: EmfieldCartesianConfig | None = None
    emfield_cylindrical: EmfieldCylindricalConfig | None = None
