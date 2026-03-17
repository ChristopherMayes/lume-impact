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
    def set_quad_k1(simulator, value, name):
        simulator.ele[name]["k1"] = value

    @transformer.getter("quad_{name}_k1")
    def get_quad_k1(simulator, name):
        return simulator.ele[name]["k1"]

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

    def set_impact_property(self, simulator: Any, name: str, value: Any) -> None:
        """Route to the most specific registered setter matching `name`."""
        func, kwargs = self._match(self._setters, name)
        if func is None:
            raise KeyError(f"No setter registered for '{name}'")
        func(simulator, value, **kwargs)

    def get_impact_property(self, simulator: Any, name: str) -> Any:
        """Route to the most specific registered getter matching `name`."""
        func, kwargs = self._match(self._getters, name)
        if func is None:
            raise KeyError(f"No getter registered for '{name}'")
        return func(simulator, **kwargs)
