import operator
import re
from typing import Any, Callable


_Route = tuple[str, re.Pattern, list[str], Callable]


def _pattern_to_regex(pattern: str) -> tuple[re.Pattern, list[str]]:
    """Convert a pattern like 'quad_{name}_k1' to a compiled regex and list of param names."""
    parts = re.split(r"\{(\w+)\}", pattern)
    regex_parts = []
    param_names = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            regex_parts.append(re.escape(part))
        else:
            param_names.append(part)
            regex_parts.append(f"(?P<{part}>.+?)")
    return re.compile("^" + "".join(regex_parts) + "$"), param_names


def _specificity(pattern: str, param_names: list[str]) -> tuple:
    """Fewer variables = more specific; longer pattern breaks ties."""
    return (len(param_names), -len(pattern))


class ImpactTransformer:
    """
    Routes get/set calls to registered handlers by pattern matching, similar to Flask/FastAPI.

    Usage
    -----
    transformer = ImpactTransformer()

    @transformer.setter("quad_{name}_k1")
    def set_quad_k1(imp, value, name):
        imp.ele[name]["k1"] = value

    @transformer.getter("quad_{name}_k1")
    def get_quad_k1(imp, name):
        return imp.ele[name]["k1"]

    transformer.set_impact_property(sim, "quad_Q1_k1", 0.5)
    transformer.get_impact_property(sim, "quad_Q1_k1")

    Pattern variables (e.g. `{name}`) are extracted from the property name and passed
    as keyword arguments to the handler. More specific patterns (fewer variables, longer
    literal portions) take priority over less specific ones.
    """

    def __init__(self):
        self._setters: list[_Route] = []
        self._getters: list[_Route] = []

    def register_setter(self, pattern: str, func: Callable) -> Callable:
        """Register a setter function for the given pattern. Returns func unchanged."""
        regex, params = _pattern_to_regex(pattern)
        self._setters.append((pattern, regex, params, func))
        self._setters.sort(key=lambda r: _specificity(r[0], r[2]))
        return func

    def register_getter(self, pattern: str, func: Callable) -> Callable:
        """Register a getter function for the given pattern. Returns func unchanged."""
        regex, params = _pattern_to_regex(pattern)
        self._getters.append((pattern, regex, params, func))
        self._getters.sort(key=lambda r: _specificity(r[0], r[2]))
        return func

    def setter(self, pattern: str) -> Callable:
        """Decorator sugar for register_setter."""

        def decorator(func: Callable) -> Callable:
            return self.register_setter(pattern, func)

        return decorator

    def getter(self, pattern: str) -> Callable:
        """Decorator sugar for register_getter."""

        def decorator(func: Callable) -> Callable:
            return self.register_getter(pattern, func)

        return decorator

    def _match(
        self, registry: list[_Route], name: str
    ) -> tuple[Callable, dict] | tuple[None, None]:
        for _, regex, _, func in registry:
            m = regex.match(name)
            if m:
                return func, m.groupdict()
        return None, None

    def set_impact_property(self, imp: Any, name: str, value: Any) -> None:
        """Route to the most specific registered setter matching `name`."""
        func, kwargs = self._match(self._setters, name)
        if func is None:
            raise KeyError(f"No setter registered for '{name}'")
        func(imp, value, **kwargs)

    def get_impact_property(self, imp: Any, name: str) -> Any:
        """Route to the most specific registered getter matching `name`."""
        func, kwargs = self._match(self._getters, name)
        if func is None:
            raise KeyError(f"No getter registered for '{name}'")
        return func(imp, **kwargs)

    def add_particle_getter(self, pattern: str) -> None:
        """Register a getter that returns ``imp.particles[name]``.

        Pattern must contain ``{name}``, which is used as the particles key.
        """
        self.register_getter(pattern, lambda imp, name: imp.particles[name])

    def add_stat_getter(self, pattern: str) -> None:
        """Register a getter that returns ``imp.stat(name)``.

        Pattern must contain ``{name}``, which is forwarded to ``stat()``.
        """
        self.register_getter(pattern, lambda imp, name: imp.stat(name))

    def add_header_getter(
        self, pattern: str, key_map: dict[str, str] | None = None
    ) -> None:
        """Register a getter that returns ``imp.header[key]``.

        Pattern must contain ``{key}``.

        Parameters
        ----------
        key_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.header`` key (e.g. ``{"sigx": "sigx(m)"}``).
        """

        def getter(imp, key, _key_map=key_map):
            return imp.header[_key_map.get(key, key) if _key_map else key]

        self.register_getter(pattern, getter)

    def add_header_setter(
        self, pattern: str, key_map: dict[str, str] | None = None
    ) -> None:
        """Register a setter that writes ``imp.header[key] = value``.

        Pattern must contain ``{key}``.

        Parameters
        ----------
        key_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.header`` key (e.g. ``{"sigx": "sigx(m)"}``).
        """

        def setter(imp, value, key, _key_map=key_map):
            operator.setitem(
                imp.header, _key_map.get(key, key) if _key_map else key, value
            )

        self.register_setter(pattern, setter)

    def add_ele_getter(
        self, pattern: str, attrib_map: dict[str, str] | None = None
    ) -> None:
        """Register a getter that returns ``imp.ele[name][attrib]``.

        Pattern must contain ``{name}`` and ``{attrib}``.

        Parameters
        ----------
        attrib_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.ele[name]`` key (e.g. ``{"rf_phase": "rf_phase_deg"}``).
        """

        def getter(imp, name, attrib, _attrib_map=attrib_map):
            return imp.ele[name][
                _attrib_map.get(attrib, attrib) if _attrib_map else attrib
            ]

        self.register_getter(pattern, getter)

    def add_ele_setter(
        self, pattern: str, attrib_map: dict[str, str] | None = None
    ) -> None:
        """Register a setter that writes ``imp.ele[name][attrib] = value``.

        Pattern must contain ``{name}`` and ``{attrib}``.

        Parameters
        ----------
        attrib_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.ele[name]`` key (e.g. ``{"rf_phase": "rf_phase_deg"}``).
        """

        def setter(imp, value, name, attrib, _attrib_map=attrib_map):
            operator.setitem(
                imp.ele[name],
                _attrib_map.get(attrib, attrib) if _attrib_map else attrib,
                value,
            )

        self.register_setter(pattern, setter)

    def add_group_getter(self, pattern: str) -> None:
        """Register a getter that returns ``imp.group[name][attrib]``.

        Pattern must contain ``{name}`` and ``{attrib}``.
        """
        self.register_getter(
            pattern,
            lambda imp, name, attrib: imp.group[name][attrib],
        )

    def add_group_setter(self, pattern: str) -> None:
        """Register a setter that writes ``imp.group[name][attrib] = value``.

        Pattern must contain ``{name}`` and ``{attrib}``.
        """
        self.register_setter(
            pattern,
            lambda imp, value, name, attrib: operator.setitem(
                imp.group[name], attrib, value
            ),
        )
