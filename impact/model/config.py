from typing import Any

from pydantic import BaseModel

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


def make_variables(imp: Any, config: VariableMappingConfig) -> list[ScalarVariable]:
    """Build a ``ScalarVariable`` for every element attribute described by *config*.

    The current value of each attribute in *imp* is used as ``default_value``.

    Parameters
    ----------
    imp : Impact
    config : VariableMappingConfig

    Returns
    -------
    list[ScalarVariable]
    """
    variables = []

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
            variable_name = config.pattern.format(
                type=ele_type, name=ele_name, attrib=attrib_token
            )

            default_value = imp.ele[ele_name].get(field_name)

            variables.append(
                ScalarVariable(
                    name=variable_name,
                    default_value=default_value,
                    unit=attr_cfg.unit,
                    read_only=False,
                )
            )

    return variables
