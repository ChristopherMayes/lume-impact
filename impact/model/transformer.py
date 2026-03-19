import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class ImpactTransformer(ABC):
    """Abstract base class for Impact-T property transformers."""

    @abstractmethod
    def get_impact_property(self, imp: Any, name: str) -> Any:
        """Return the current value of the named property from *imp*."""

    @abstractmethod
    def set_impact_property(self, imp: Any, name: str, value: Any) -> None:
        """Write *value* to the named property on *imp*."""


# ------------------------------------------------------------------

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


def _compile_re(p: str | re.Pattern | None) -> re.Pattern | None:
    """Compile *p* to a Pattern if it is a string; pass through compiled patterns and None."""
    if p is None:
        return None
    return re.compile(p) if isinstance(p, str) else p


@dataclass(slots=True)
class _EleRoute:
    """Match criteria and handler for an element-specific getter or setter."""

    ele_type_re: re.Pattern | None
    name_re: re.Pattern | None
    mapped_name_re: re.Pattern | None
    attrib_re: re.Pattern | None
    mapped_attrib_re: re.Pattern | None
    func: Callable

    def matches(
        self,
        ele_type: str,
        name: str,
        mapped_name: str,
        attrib: str,
        mapped_attrib: str,
    ) -> bool:
        return (
            (self.ele_type_re is None or self.ele_type_re.search(ele_type))
            and (self.name_re is None or self.name_re.search(name))
            and (self.mapped_name_re is None or self.mapped_name_re.search(mapped_name))
            and (self.attrib_re is None or self.attrib_re.search(attrib))
            and (
                self.mapped_attrib_re is None
                or self.mapped_attrib_re.search(mapped_attrib)
            )
        )


class RoutingImpactTransformer(ImpactTransformer):
    """
    Routes get/set calls to registered handlers by pattern matching, similar to Flask/FastAPI.

    Usage
    -----
    transformer = RoutingImpactTransformer()

    @transformer.setter(pattern="quad_{name}_k1")
    def set_quad_k1(imp, value, name):
        imp.ele[name]["k1"] = value

    @transformer.getter(pattern="quad_{name}_k1")
    def get_quad_k1(imp, name):
        return imp.ele[name]["k1"]

    transformer.set_impact_property(sim, "quad_Q1_k1", 0.5)
    transformer.get_impact_property(sim, "quad_Q1_k1")

    Pattern variables (e.g. `{name}`) are extracted from the property name and passed
    as keyword arguments to the handler. More specific patterns (fewer variables, longer
    literal portions) take priority over less specific ones.

    Element-specific handlers
    -------------------------
    Element getters and setters registered via ``add_ele_getter`` / ``add_ele_setter``
    dispatch through a separate queue of element-level handlers before falling back to
    the default ``imp.ele[name][attrib]`` behaviour. Register handlers with
    ``register_ele_getter`` / ``register_ele_setter`` (or their decorator equivalents
    ``ele_getter`` / ``ele_setter``). Handlers are matched in registration order
    (most-recently registered first) against four optional regex criteria:

    * ``ele_type``    -- element type from Impact-T (``imp.ele[mapped_name].get("type", "")``)
    * ``name``        -- pre-mapped element name (the ``{name}`` token)
    * ``mapped_name`` -- post-mapped element name (key into ``imp.ele``)
    * ``attrib``      -- attribute token (the ``{attrib}`` token)
    * ``mapped_attrib`` -- post-mapped attribute (after ``ele_attrib_map``)

    Element getter handler signature: ``func(imp, name, mapped_name, attrib) -> value``
    Element setter handler signature: ``func(imp, value, name, mapped_name, attrib)``
    """

    def __init__(
        self,
        ele_pattern=None,
        ele_regex=None,
        ele_name_map: dict[str, str] | None = None,
        ele_attrib_map: dict[str, str] | None = None,
    ):
        if ele_name_map is None:
            ele_name_map = {}
        if ele_attrib_map is None:
            ele_attrib_map = {}

        self._setters: list[_Route] = []
        self._getters: list[_Route] = []
        self._ele_setters: list[_EleRoute] = []
        self._ele_getters: list[_EleRoute] = []
        self._ele_name_map = ele_name_map
        self._ele_attrib_map = ele_attrib_map

        if ele_pattern is not None or ele_regex is not None:
            self.register_getter(
                self._ele_getter,
                pattern=ele_pattern,
                regex=ele_regex,
            )
            self.register_setter(
                self._ele_setter,
                pattern=ele_pattern,
                regex=ele_regex,
            )

    # ------------------------------------------------------------------
    # General register / decorator API

    def register_setter(
        self,
        func: Callable,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> Callable:
        """Register a setter function. Exactly one of *pattern* or *regex* must be given.

        Parameters
        ----------
        pattern :
            Template string like ``'quad_{name}_k1'``; ``{var}`` tokens become
            named regex groups and are passed as kwargs to *func*.
        regex :
            A raw regex string or compiled ``re.Pattern``. Named groups are
            extracted and passed as kwargs to *func*.
        """
        if (pattern is None) == (regex is None):
            raise ValueError("Exactly one of 'pattern' or 'regex' must be provided")
        if pattern is not None:
            compiled, params = _pattern_to_regex(pattern)
            sort_key = pattern
        else:
            compiled = re.compile(regex) if isinstance(regex, str) else regex
            params = list(compiled.groupindex.keys())
            sort_key = compiled.pattern
        self._setters.append((sort_key, compiled, params, func))
        self._setters.sort(key=lambda r: _specificity(r[0], r[2]))
        return func

    def register_getter(
        self,
        func: Callable,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> Callable:
        """Register a getter function. Exactly one of *pattern* or *regex* must be given.

        Parameters
        ----------
        pattern :
            Template string like ``'quad_{name}_k1'``; ``{var}`` tokens become
            named regex groups and are passed as kwargs to *func*.
        regex :
            A raw regex string or compiled ``re.Pattern``. Named groups are
            extracted and passed as kwargs to *func*.
        """
        if (pattern is None) == (regex is None):
            raise ValueError("Exactly one of 'pattern' or 'regex' must be provided")
        if pattern is not None:
            compiled, params = _pattern_to_regex(pattern)
            sort_key = pattern
        else:
            compiled = re.compile(regex) if isinstance(regex, str) else regex
            params = list(compiled.groupindex.keys())
            sort_key = compiled.pattern
        self._getters.append((sort_key, compiled, params, func))
        self._getters.sort(key=lambda r: _specificity(r[0], r[2]))
        return func

    def setter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> Callable:
        """Decorator sugar for register_setter."""

        def decorator(func: Callable) -> Callable:
            return self.register_setter(func, pattern=pattern, regex=regex)

        return decorator

    def getter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> Callable:
        """Decorator sugar for register_getter."""

        def decorator(func: Callable) -> Callable:
            return self.register_getter(func, pattern=pattern, regex=regex)

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

    # ------------------------------------------------------------------
    # Element-specific register / decorator API

    def register_ele_getter(
        self,
        func: Callable,
        *,
        ele_type: str | re.Pattern | None = ".*",
        name: str | re.Pattern | None = ".*",
        mapped_name: str | re.Pattern | None = ".*",
        attrib: str | re.Pattern | None = ".*",
        mapped_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Register an element getter. Most-recently registered handler wins.

        All criteria default to ``".*"`` (match anything); pass a narrower regex
        to restrict matching.

        Parameters
        ----------
        ele_type :
            Regex matched against the element type reported by Impact-T
            (``imp.ele[mapped_name].get("type", "")``).
        name :
            Regex matched against the pre-mapped element name (``{name}`` token).
        mapped_name :
            Regex matched against the post-mapped element name (key in ``imp.ele``).
        attrib :
            Regex matched against the pre-mapped attribute token (``{attrib}`` token).
        mapped_attrib :
            Regex matched against the post-mapped attribute (after ``ele_attrib_map``).

        Handler signature: ``func(imp, name, mapped_name, attrib, **kwargs) -> value``
        """
        self._ele_getters.insert(
            0,
            _EleRoute(
                ele_type_re=_compile_re(ele_type),
                name_re=_compile_re(name),
                mapped_name_re=_compile_re(mapped_name),
                attrib_re=_compile_re(attrib),
                mapped_attrib_re=_compile_re(mapped_attrib),
                func=func,
            ),
        )
        return func

    def register_ele_setter(
        self,
        func: Callable,
        *,
        ele_type: str | re.Pattern | None = ".*",
        name: str | re.Pattern | None = ".*",
        mapped_name: str | re.Pattern | None = ".*",
        attrib: str | re.Pattern | None = ".*",
        mapped_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Register an element setter. Most-recently registered handler wins.

        All criteria default to ``".*"`` (match anything); pass a narrower regex
        to restrict matching.

        Parameters
        ----------
        ele_type :
            Regex matched against the element type reported by Impact-T.
        name :
            Regex matched against the pre-mapped element name.
        mapped_name :
            Regex matched against the post-mapped element name.
        attrib :
            Regex matched against the pre-mapped attribute token.
        mapped_attrib :
            Regex matched against the post-mapped attribute (after ``ele_attrib_map``).

        Handler signature: ``func(imp, value, name, mapped_name, attrib, **kwargs)``
        """
        self._ele_setters.insert(
            0,
            _EleRoute(
                ele_type_re=_compile_re(ele_type),
                name_re=_compile_re(name),
                mapped_name_re=_compile_re(mapped_name),
                attrib_re=_compile_re(attrib),
                mapped_attrib_re=_compile_re(mapped_attrib),
                func=func,
            ),
        )
        return func

    def ele_getter(
        self,
        *,
        ele_type: str | re.Pattern | None = ".*",
        name: str | re.Pattern | None = ".*",
        mapped_name: str | re.Pattern | None = ".*",
        attrib: str | re.Pattern | None = ".*",
        mapped_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Decorator sugar for register_ele_getter."""

        def decorator(func: Callable) -> Callable:
            return self.register_ele_getter(
                func,
                ele_type=ele_type,
                name=name,
                mapped_name=mapped_name,
                attrib=attrib,
                mapped_attrib=mapped_attrib,
            )

        return decorator

    def ele_setter(
        self,
        *,
        ele_type: str | re.Pattern | None = ".*",
        name: str | re.Pattern | None = ".*",
        mapped_name: str | re.Pattern | None = ".*",
        attrib: str | re.Pattern | None = ".*",
        mapped_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Decorator sugar for register_ele_setter."""

        def decorator(func: Callable) -> Callable:
            return self.register_ele_setter(
                func,
                ele_type=ele_type,
                name=name,
                mapped_name=mapped_name,
                attrib=attrib,
                mapped_attrib=mapped_attrib,
            )

        return decorator

    # ------------------------------------------------------------------
    # Element dispatch methods (registered by add_ele_getter / add_ele_setter)

    def _ele_getter(
        self,
        imp: Any,
        name: str,
        attrib: str,
        **kwargs,
    ) -> Any:
        """Dispatch a getter call through ``_ele_getters``, falling back to ``imp.ele[mapped_name][mapped_attrib]``."""
        mapped_name = self._ele_name_map.get(name, name)
        mapped_attrib = self._ele_attrib_map.get(attrib, attrib)
        ele_type = imp.ele.get(mapped_name, {}).get("type", "")
        for route in self._ele_getters:
            if route.matches(ele_type, name, mapped_name, attrib, mapped_attrib):
                return route.func(imp, name, mapped_name, attrib, **kwargs)
        return imp.ele[mapped_name][mapped_attrib]

    def _ele_setter(
        self,
        imp: Any,
        value: Any,
        name: str,
        attrib: str,
        **kwargs,
    ) -> None:
        """Dispatch a setter call through ``_ele_setters``, falling back to ``imp.ele[mapped_name][mapped_attrib] = value``."""
        mapped_name = self._ele_name_map.get(name, name)
        mapped_attrib = self._ele_attrib_map.get(attrib, attrib)
        ele_type = imp.ele.get(mapped_name, {}).get("type", "")
        for route in self._ele_setters:
            if route.matches(ele_type, name, mapped_name, attrib, mapped_attrib):
                route.func(imp, value, name, mapped_name, attrib, **kwargs)
                return
        operator.setitem(imp.ele[mapped_name], mapped_attrib, value)

    # ------------------------------------------------------------------
    # Convenience add_* methods

    def add_particle_getter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> None:
        """Register a getter that returns ``imp.particles[name]``.

        The pattern/regex must contain a ``name`` named group.
        """
        self.register_getter(
            lambda imp, name, **kwargs: imp.particles[name],
            pattern=pattern,
            regex=regex,
        )

    def add_stat_getter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> None:
        """Register a getter that returns ``imp.stat(name)``.

        The pattern/regex must contain a ``name`` named group.
        """
        self.register_getter(
            lambda imp, name, **kwargs: imp.stat(name), pattern=pattern, regex=regex
        )

    def add_header_getter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
        key_map: dict[str, str] | None = None,
    ) -> None:
        """Register a getter that returns ``imp.header[key]``.

        The pattern/regex must contain a ``key`` named group.

        Parameters
        ----------
        key_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.header`` key (e.g. ``{"sigx": "sigx(m)"}``).
        """

        def getter(imp, key, _key_map=key_map, **kwargs):
            return imp.header[_key_map.get(key, key) if _key_map else key]

        self.register_getter(getter, pattern=pattern, regex=regex)

    def add_header_setter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
        key_map: dict[str, str] | None = None,
    ) -> None:
        """Register a setter that writes ``imp.header[key] = value``.

        The pattern/regex must contain a ``key`` named group.

        Parameters
        ----------
        key_map :
            Optional mapping from the token extracted from the variable name to the
            actual ``imp.header`` key (e.g. ``{"sigx": "sigx(m)"}``).
        """

        def setter(imp, value, key, _key_map=key_map, **kwargs):
            operator.setitem(
                imp.header, _key_map.get(key, key) if _key_map else key, value
            )

        self.register_setter(setter, pattern=pattern, regex=regex)

    def add_group_getter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> None:
        """Register a getter that returns ``imp.group[name][attrib]``.

        The pattern/regex must contain ``name`` and ``attrib`` named groups.
        """
        self.register_getter(
            lambda imp, name, attrib, **kwargs: imp.group[name][attrib],
            pattern=pattern,
            regex=regex,
        )

    def add_group_setter(
        self,
        *,
        pattern: str | None = None,
        regex: str | re.Pattern | None = None,
    ) -> None:
        """Register a setter that writes ``imp.group[name][attrib] = value``.

        The pattern/regex must contain ``name`` and ``attrib`` named groups.
        """
        self.register_setter(
            lambda imp, value, name, attrib, **kwargs: operator.setitem(
                imp.group[name], attrib, value
            ),
            pattern=pattern,
            regex=regex,
        )
