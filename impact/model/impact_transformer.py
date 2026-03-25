import operator
import re
from typing import Any

from impact.model.transformer.ele_routing import RoutingEleTransformer


class ImpactTransformer(RoutingEleTransformer):
    """
    Impact-T specific routing transformer with convenience registration helpers.

    Inherits pattern-based routing from ``RoutingTransformer`` and element-specific
    routing from ``RoutingEleTransformer``. Adds helpers for registering getters and
    setters that target common Impact-T data structures (particles, stats, header,
    group, and elements).
    """

    def get_ele_type(self, tool: Any, tool_name: str) -> str:
        return tool.ele.get(tool_name, {}).get("type", "")

    def default_ele_getter(
        self, tool, control_name, tool_name, control_attrib, tool_attrib
    ):
        return tool.ele[tool_name][tool_attrib]

    def default_ele_setter(
        self, tool, value, control_name, tool_name, control_attrib, tool_attrib
    ):
        operator.setitem(tool.ele[tool_name], tool_attrib, value)

    def get_impact_property(self, imp: Any, name: str) -> Any:
        """Return the current value of the named property from *imp*."""
        return self.get_property(imp, name)

    def set_impact_property(self, imp: Any, name: str, value: Any) -> None:
        """Write *value* to the named property on *imp*."""
        self.set_property(imp, name, value)

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
