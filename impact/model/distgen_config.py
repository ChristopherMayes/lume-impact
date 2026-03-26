import logging
from typing import Any
from pydantic import BaseModel

from lume.variables import ScalarVariable

logger = logging.getLogger(__name__)


class DistgenParamConfig(BaseModel):
    """Config for a single distgen parameter.

    Parameters
    ----------
    alias : str, optional
        Override the variable name token.
    distgen_param : str, optional
        Override the distgen parameter key (e.g. ``"sigma_t"`` instead of derived name).
    unit : str, optional
        Physical unit string passed to ``ScalarVariable``.
    read_only : bool, optional
        Whether the variable is read-only.  Defaults to ``False``.
    exclude : bool, optional
        Exclude this parameter entirely.  Defaults to ``False``.
    """

    alias: str | None = None
    distgen_param: str | None = None
    unit: str | None = None
    read_only: bool = False
    exclude: bool = False


# ------------------------------------------------------------------
# Distribution type configs
# Set a field to DistgenParamConfig() to include it; None to exclude.
# ------------------------------------------------------------------


class GaussianDistConfig(BaseModel):
    sigma: DistgenParamConfig | None = DistgenParamConfig()
    avg: DistgenParamConfig | None = None
    n_sigma_cutoff: DistgenParamConfig | None = None
    n_sigma_cutoff_left: DistgenParamConfig | None = None
    n_sigma_cutoff_right: DistgenParamConfig | None = None


class UniformDistConfig(BaseModel):
    min: DistgenParamConfig | None = DistgenParamConfig()
    max: DistgenParamConfig | None = DistgenParamConfig()
    avg: DistgenParamConfig | None = None
    sigma: DistgenParamConfig | None = None


class SuperGaussianDistConfig(BaseModel):
    sigma: DistgenParamConfig | None = DistgenParamConfig()
    avg: DistgenParamConfig | None = None
    p: DistgenParamConfig | None = DistgenParamConfig()
    n_sigma_cutoff: DistgenParamConfig | None = None


class TukeyDistConfig(BaseModel):
    length: DistgenParamConfig | None = DistgenParamConfig()
    ratio: DistgenParamConfig | None = DistgenParamConfig()


class RadialGaussianDistConfig(BaseModel):
    sigma_xy: DistgenParamConfig | None = DistgenParamConfig()
    n_sigma_cutoff: DistgenParamConfig | None = None
    n_sigma_cutoff_left: DistgenParamConfig | None = None
    n_sigma_cutoff_right: DistgenParamConfig | None = None
    truncation_radius: DistgenParamConfig | None = None
    truncation_fraction: DistgenParamConfig | None = None


class RadialUniformDistConfig(BaseModel):
    max_r: DistgenParamConfig | None = DistgenParamConfig()
    min_r: DistgenParamConfig | None = None


class RadialSuperGaussianDistConfig(BaseModel):
    sigma_xy: DistgenParamConfig | None = DistgenParamConfig()
    p: DistgenParamConfig | None = DistgenParamConfig()


class RadialTukeyDistConfig(BaseModel):
    length: DistgenParamConfig | None = DistgenParamConfig()
    ratio: DistgenParamConfig | None = DistgenParamConfig()


# ------------------------------------------------------------------
# Start configs
# ------------------------------------------------------------------


class CathodeStartConfig(BaseModel):
    MTE: DistgenParamConfig | None = DistgenParamConfig()


class StartConfig(BaseModel):
    cathode: CathodeStartConfig | None = CathodeStartConfig()


# ------------------------------------------------------------------
# Per-slot dist configs (one per coordinate slot in gen.input)
# ------------------------------------------------------------------


class RDistConfig(BaseModel):
    radial_gaussian: RadialGaussianDistConfig | None = RadialGaussianDistConfig()
    radial_uniform: RadialUniformDistConfig | None = RadialUniformDistConfig()
    radial_super_gaussian: RadialSuperGaussianDistConfig | None = None
    radial_tukey: RadialTukeyDistConfig | None = None


class TDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()
    super_gaussian: SuperGaussianDistConfig | None = None
    tukey: TukeyDistConfig | None = TukeyDistConfig()


class XDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()
    super_gaussian: SuperGaussianDistConfig | None = None


class YDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()
    super_gaussian: SuperGaussianDistConfig | None = None


class ZDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()
    super_gaussian: SuperGaussianDistConfig | None = None


class PxDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()


class PyDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()


class PzDistConfig(BaseModel):
    gaussian: GaussianDistConfig | None = GaussianDistConfig()
    uniform: UniformDistConfig | None = UniformDistConfig()


# ------------------------------------------------------------------
# Mapping from slot name → coordinate suffix used for 1D dist params
# (e.g. sigma_t for t_dist, sigma_x for x_dist; radial has no suffix)
# ------------------------------------------------------------------

_COORD_FROM_DIST: dict[str, str | None] = {
    "r_dist": None,
    "t_dist": "t",
    "x_dist": "x",
    "y_dist": "y",
    "z_dist": "z",
    "px_dist": "px",
    "py_dist": "py",
    "pz_dist": "pz",
}


# ------------------------------------------------------------------
# Top-level input config
# ------------------------------------------------------------------


class DistgenInputConfig(BaseModel):
    """Config for the distgen generator input.

    Each field maps to a slot in ``gen.input``.  Setting a field to
    ``None`` suppresses variable discovery for that slot.

    Parameters
    ----------
    pattern : str
        Pattern template for variable names.  Must contain ``{key}``
        which is replaced with the colon-separated distgen path.
    """

    pattern: str = "distgen/{key}"
    n_particle: DistgenParamConfig | None = DistgenParamConfig()
    total_charge: DistgenParamConfig | None = DistgenParamConfig()
    start: StartConfig | None = StartConfig()
    r_dist: RDistConfig | None = RDistConfig()
    t_dist: TDistConfig | None = TDistConfig()
    x_dist: XDistConfig | None = XDistConfig()
    y_dist: YDistConfig | None = YDistConfig()
    z_dist: ZDistConfig | None = ZDistConfig()
    px_dist: PxDistConfig | None = PxDistConfig()
    py_dist: PyDistConfig | None = PyDistConfig()
    pz_dist: PzDistConfig | None = PzDistConfig()


class DistgenVariableMappingConfig(BaseModel):
    """Top-level config for building distgen variables and transformers."""

    inputs: DistgenInputConfig | None = DistgenInputConfig()


# ------------------------------------------------------------------
# Variable mapping dataclass
# ------------------------------------------------------------------


class DistgenVariableMapping:
    """Holds a distgen variable and the info needed to get/set it."""

    __slots__ = ("var", "token", "key", "has_units")

    def __init__(
        self,
        var: ScalarVariable,
        token: str,
        key: str,
        has_units: bool,
    ):
        self.var = var
        self.token = token
        self.key = key
        self.has_units = has_units


class DistgenVariableMappings:
    def __init__(self, input_mappings: list[DistgenVariableMapping]):
        self.input_mappings = input_mappings

    @property
    def all_vars(self) -> list[ScalarVariable]:
        return [m.var for m in self.input_mappings]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_param_key(field: str, coord: str | None) -> str:
    """Return the distgen parameter key for a 1D dist field + coord.

    For radial dists (coord=None) the field IS the key.
    For 1D dists, params that carry the coordinate get ``{field}_{coord}``
    (e.g. ``sigma_t``); coord-independent params use the bare field name.
    """
    _COORD_PARAMS = {"sigma", "avg", "min", "max", "scale", "lambda"}
    if coord is None:
        return field
    if field in _COORD_PARAMS:
        return f"{field}_{coord}"
    return field


def _is_quantity(raw: Any) -> bool:
    return isinstance(raw, dict) and "value" in raw


def _make_var(
    name: str,
    param_cfg: DistgenParamConfig,
    default_unit: str | None,
    read_only: bool,
) -> ScalarVariable:
    unit = param_cfg.unit if param_cfg.unit is not None else default_unit
    return ScalarVariable(
        name=name,
        default_value=None,
        unit=unit,
        read_only=read_only or param_cfg.read_only,
    )


def _process_dist_config(
    gen_input: dict,
    slot: str,
    dist_type: str,
    type_cfg: BaseModel,
    coord: str | None,
    pattern: str,
    mappings: list[DistgenVariableMapping],
) -> None:
    """Walk a distribution type config and emit variable mappings."""
    dist_params = gen_input.get(slot, {})
    for field in type_cfg.model_fields:
        val = getattr(type_cfg, field)
        if val is None or not isinstance(val, DistgenParamConfig) or val.exclude:
            continue
        distgen_key = val.distgen_param or _resolve_param_key(field, coord)
        raw = dist_params.get(distgen_key)
        if raw is None:
            continue
        has_units = _is_quantity(raw)
        default_unit = raw.get("units") if has_units else None
        token = val.alias or field
        var_name = pattern.replace("{key}", f"{slot}/{dist_type}/{token}")
        full_key = f"{slot}:{distgen_key}"
        if has_units:
            full_key += ":value"
        var = _make_var(var_name, val, default_unit, read_only=False)
        mappings.append(
            DistgenVariableMapping(
                var, token=var_name, key=full_key, has_units=has_units
            )
        )


def _process_slot_config(
    gen_input: dict,
    slot: str,
    slot_cfg: BaseModel,
    coord: str | None,
    pattern: str,
    mappings: list[DistgenVariableMapping],
) -> None:
    """Walk a per-slot config and emit variable mappings for the active dist type."""
    dist_params = gen_input.get(slot, {})
    if not dist_params:
        return
    active_type = dist_params.get("type", "")
    for type_field in type(slot_cfg).model_fields:
        cfg = getattr(slot_cfg, type_field)
        if cfg is None or not isinstance(cfg, BaseModel):
            continue
        if active_type and type_field != active_type:
            continue
        _process_dist_config(gen_input, slot, type_field, cfg, coord, pattern, mappings)


def _process_start_config(
    gen_input: dict,
    start_cfg: StartConfig,
    pattern: str,
    mappings: list[DistgenVariableMapping],
) -> None:
    """Walk the start config and emit variable mappings for the active start type."""
    start_params = gen_input.get("start", {})
    if not start_params:
        return
    active_type = start_params.get("type", "")
    for type_field in type(start_cfg).model_fields:
        cfg = getattr(start_cfg, type_field)
        if cfg is None or not isinstance(cfg, BaseModel):
            continue
        if active_type and type_field != active_type:
            continue
        for field in type(cfg).model_fields:
            param_cfg = getattr(cfg, field)
            if (
                param_cfg is None
                or not isinstance(param_cfg, DistgenParamConfig)
                or param_cfg.exclude
            ):
                continue
            distgen_key = param_cfg.distgen_param or field
            raw = start_params.get(distgen_key)
            if raw is None:
                continue
            has_units = _is_quantity(raw)
            default_unit = raw.get("units") if has_units else None
            token = param_cfg.alias or field
            var_name = pattern.replace("{key}", f"start/{type_field}/{token}")
            full_key = f"start:{distgen_key}"
            if has_units:
                full_key += ":value"
            var = _make_var(var_name, param_cfg, default_unit, read_only=False)
            mappings.append(
                DistgenVariableMapping(
                    var, token=var_name, key=full_key, has_units=has_units
                )
            )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def make_variables(
    gen: Any,
    config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
) -> DistgenVariableMappings:
    """Build variable mappings from a distgen Generator and config.

    Parameters
    ----------
    gen :
        A distgen ``Generator`` instance.
    config :
        Variable mapping configuration.  Defaults to auto-discovering all
        parameters present in ``gen.input``.

    Returns
    -------
    DistgenVariableMappings
    """
    mappings: list[DistgenVariableMapping] = []
    inp_cfg = config.inputs
    if inp_cfg is None:
        return DistgenVariableMappings(mappings)

    gen_input = gen.input
    pattern = inp_cfg.pattern

    # Top-level scalar params
    for field in ("n_particle", "total_charge"):
        param_cfg: DistgenParamConfig | None = getattr(inp_cfg, field)
        if param_cfg is None or param_cfg.exclude:
            continue
        raw = gen_input.get(field)
        if raw is None:
            continue
        has_units = _is_quantity(raw)
        default_unit = raw.get("units") if has_units else None
        token = param_cfg.alias or field
        var_name = pattern.replace("{key}", token)
        full_key = field
        if has_units:
            full_key += ":value"
        var = _make_var(var_name, param_cfg, default_unit, read_only=False)
        mappings.append(
            DistgenVariableMapping(
                var, token=var_name, key=full_key, has_units=has_units
            )
        )

    # Start
    if inp_cfg.start is not None:
        _process_start_config(gen_input, inp_cfg.start, pattern, mappings)

    # Distribution slots
    for slot in _COORD_FROM_DIST:
        slot_cfg = getattr(inp_cfg, slot)
        if slot_cfg is None:
            continue
        coord = _COORD_FROM_DIST[slot]
        _process_slot_config(gen_input, slot, slot_cfg, coord, pattern, mappings)

    return DistgenVariableMappings(mappings)


def make_transformer(
    var_mappings: DistgenVariableMappings,
    config: DistgenVariableMappingConfig = DistgenVariableMappingConfig(),
    transformer=None,
):
    """Build a transformer for a distgen Generator.

    Parameters
    ----------
    var_mappings :
        Pre-computed variable mappings from ``make_variables``.
    config :
        Variable mapping configuration.
    transformer :
        Optional existing transformer to extend.

    Returns
    -------
    RoutingTransformer
    """
    from impact.model.transformer.routing import RoutingTransformer

    if transformer is None:
        _trans = RoutingTransformer()
    else:
        _trans = transformer

    key_map = {m.token: m.key for m in var_mappings.input_mappings}

    _trans.register_getter(
        lambda gen, key, _m=key_map, **kwargs: gen[_m[key]],
        pattern=config.inputs.pattern if config.inputs else "distgen/{key}",
    )
    _trans.register_setter(
        lambda gen, value, key, _m=key_map, **kwargs: gen.__setitem__(_m[key], value),
        pattern=config.inputs.pattern if config.inputs else "distgen/{key}",
    )

    return _trans
