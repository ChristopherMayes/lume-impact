import dataclasses
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from impact.model.transformer.routing import RoutingTransformer


def _compile_re(p: str | re.Pattern | None) -> re.Pattern | None:
    """Compile *p* to a Pattern if it is a string; pass through compiled patterns and None."""
    if p is None:
        return None
    return re.compile(p) if isinstance(p, str) else p


@dataclass(slots=True)
class _EleRoute:
    """Match criteria and handler for an element-specific getter or setter."""

    ele_type_re: re.Pattern | None
    control_name_re: re.Pattern | None
    tool_name_re: re.Pattern | None
    control_attrib_re: re.Pattern | None
    tool_attrib_re: re.Pattern | None
    func: Callable

    def matches(
        self,
        ele_type: str,
        control_name: str,
        tool_name: str,
        control_attrib: str,
        tool_attrib: str,
    ) -> bool:
        return (
            (self.ele_type_re is None or self.ele_type_re.search(ele_type))
            and (
                self.control_name_re is None
                or self.control_name_re.search(control_name)
            )
            and (self.tool_name_re is None or self.tool_name_re.search(tool_name))
            and (
                self.control_attrib_re is None
                or self.control_attrib_re.search(control_attrib)
            )
            and (self.tool_attrib_re is None or self.tool_attrib_re.search(tool_attrib))
        )


class RoutingEleTransformer(RoutingTransformer):
    """
    Extends RoutingTransformer with element-specific routing.

    Element getters and setters registered via ``register_ele_getter`` /
    ``register_ele_setter`` (or their decorator equivalents ``ele_getter`` /
    ``ele_setter``) dispatch through a separate queue of element-level handlers.
    Handlers are matched in registration order (most-recently registered first)
    against five optional regex criteria:

    * ``ele_type``       -- element type (resolved via ``get_ele_type``)
    * ``control_name``   -- pre-mapped element name (the ``{name}`` token from the pattern)
    * ``tool_name``      -- post-mapped element name (key into the element store)
    * ``control_attrib`` -- attribute token (the ``{attrib}`` token from the pattern)
    * ``tool_attrib``    -- post-mapped attribute (after ``ele_attrib_map``)

    Element getter handler signature:
        ``func(tool, control_name, tool_name, control_attrib, tool_attrib, **kwargs) -> value``
    Element setter handler signature:
        ``func(tool, value, control_name, tool_name, control_attrib, tool_attrib, **kwargs)``

    Subclasses must implement ``get_ele_type``.
    """

    @abstractmethod
    def get_ele_type(self, tool: Any, tool_name: str) -> str:
        """Return the element type string for the element identified by *tool_name*."""

    def __init__(
        self,
        ele_pattern=None,
        ele_regex=None,
        ele_name_map: dict[str, str] | None = None,
        ele_attrib_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self._ele_setters: list[_EleRoute] = []
        self._ele_getters: list[_EleRoute] = []
        self._ele_name_map = ele_name_map or {}
        self._ele_attrib_map = ele_attrib_map or {}

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
    # Element-specific register / decorator API

    def register_ele_getter(
        self,
        func: Callable,
        *,
        ele_type: str | re.Pattern | None = ".*",
        control_name: str | re.Pattern | None = ".*",
        tool_name: str | re.Pattern | None = ".*",
        control_attrib: str | re.Pattern | None = ".*",
        tool_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Register an element getter. Most-recently registered handler wins.

        All criteria default to ``".*"`` (match anything); pass a narrower regex
        to restrict matching.

        Parameters
        ----------
        ele_type :
            Regex matched against the element type (via ``get_ele_type``).
        control_name :
            Regex matched against the pre-mapped element name (``{name}`` token).
        tool_name :
            Regex matched against the post-mapped element name (key in the element store).
        control_attrib :
            Regex matched against the pre-mapped attribute token (``{attrib}`` token).
        tool_attrib :
            Regex matched against the post-mapped attribute (after ``ele_attrib_map``).

        Handler signature:
            ``func(tool, control_name, tool_name, control_attrib, tool_attrib, **kwargs) -> value``
        """
        self._ele_getters.insert(
            0,
            _EleRoute(
                ele_type_re=_compile_re(ele_type),
                control_name_re=_compile_re(control_name),
                tool_name_re=_compile_re(tool_name),
                control_attrib_re=_compile_re(control_attrib),
                tool_attrib_re=_compile_re(tool_attrib),
                func=func,
            ),
        )
        return func

    def register_ele_setter(
        self,
        func: Callable,
        *,
        ele_type: str | re.Pattern | None = ".*",
        control_name: str | re.Pattern | None = ".*",
        tool_name: str | re.Pattern | None = ".*",
        control_attrib: str | re.Pattern | None = ".*",
        tool_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Register an element setter. Most-recently registered handler wins.

        All criteria default to ``".*"`` (match anything); pass a narrower regex
        to restrict matching.

        Parameters
        ----------
        ele_type :
            Regex matched against the element type.
        control_name :
            Regex matched against the pre-mapped element name.
        tool_name :
            Regex matched against the post-mapped element name.
        control_attrib :
            Regex matched against the pre-mapped attribute token.
        tool_attrib :
            Regex matched against the post-mapped attribute (after ``ele_attrib_map``).

        Handler signature:
            ``func(tool, value, control_name, tool_name, control_attrib, tool_attrib, **kwargs)``
        """
        self._ele_setters.insert(
            0,
            _EleRoute(
                ele_type_re=_compile_re(ele_type),
                control_name_re=_compile_re(control_name),
                tool_name_re=_compile_re(tool_name),
                control_attrib_re=_compile_re(control_attrib),
                tool_attrib_re=_compile_re(tool_attrib),
                func=func,
            ),
        )
        return func

    def ele_getter(
        self,
        *,
        ele_type: str | re.Pattern | None = ".*",
        control_name: str | re.Pattern | None = ".*",
        tool_name: str | re.Pattern | None = ".*",
        control_attrib: str | re.Pattern | None = ".*",
        tool_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Decorator sugar for register_ele_getter."""

        def decorator(func: Callable) -> Callable:
            return self.register_ele_getter(
                func,
                ele_type=ele_type,
                control_name=control_name,
                tool_name=tool_name,
                control_attrib=control_attrib,
                tool_attrib=tool_attrib,
            )

        return decorator

    def ele_setter(
        self,
        *,
        ele_type: str | re.Pattern | None = ".*",
        control_name: str | re.Pattern | None = ".*",
        tool_name: str | re.Pattern | None = ".*",
        control_attrib: str | re.Pattern | None = ".*",
        tool_attrib: str | re.Pattern | None = ".*",
    ) -> Callable:
        """Decorator sugar for register_ele_setter."""

        def decorator(func: Callable) -> Callable:
            return self.register_ele_setter(
                func,
                ele_type=ele_type,
                control_name=control_name,
                tool_name=tool_name,
                control_attrib=control_attrib,
                tool_attrib=tool_attrib,
            )

        return decorator

    # ------------------------------------------------------------------
    # Default element getter / setter (override in subclasses for fallback behaviour)

    def default_ele_getter(
        self,
        tool: Any,
        control_name: str,
        tool_name: str,
        control_attrib: str,
        tool_attrib: str,
    ) -> Any:
        """Fallback getter called when no registered route matches.

        Return ``NotImplemented`` (the default) to have the dispatcher raise a
        ``KeyError``. Override in a subclass to provide a real fallback.
        """
        return NotImplemented

    def default_ele_setter(
        self,
        tool: Any,
        value: Any,
        control_name: str,
        tool_name: str,
        control_attrib: str,
        tool_attrib: str,
    ) -> Any:
        """Fallback setter called when no registered route matches.

        Return ``NotImplemented`` (the default) to have the dispatcher raise a
        ``KeyError``. Override in a subclass to provide a real fallback.
        """
        return NotImplemented

    # ------------------------------------------------------------------
    # Element dispatch methods

    def _ele_getter(
        self,
        tool: Any,
        name: str,
        attrib: str,
        **kwargs,
    ) -> Any:
        """Dispatch a getter call through ``_ele_getters``, then ``default_ele_getter``."""
        control_name = name
        control_attrib = attrib
        tool_name = self._ele_name_map.get(control_name, control_name)
        tool_attrib = self._ele_attrib_map.get(control_attrib, control_attrib)
        ele_type = self.get_ele_type(tool, tool_name)
        for route in self._ele_getters:
            if route.matches(
                ele_type, control_name, tool_name, control_attrib, tool_attrib
            ):
                return route.func(
                    tool, control_name, tool_name, control_attrib, tool_attrib, **kwargs
                )
        result = self.default_ele_getter(
            tool, control_name, tool_name, control_attrib, tool_attrib
        )
        if result is NotImplemented:
            raise KeyError(
                f"No element getter matched for control_name={control_name!r}, "
                f"control_attrib={control_attrib!r}, ele_type={ele_type!r}"
            )
        return result

    def _ele_setter(
        self,
        tool: Any,
        value: Any,
        name: str,
        attrib: str,
        **kwargs,
    ) -> None:
        """Dispatch a setter call through ``_ele_setters``, then ``default_ele_setter``."""
        control_name = name
        control_attrib = attrib
        tool_name = self._ele_name_map.get(control_name, control_name)
        tool_attrib = self._ele_attrib_map.get(control_attrib, control_attrib)
        ele_type = self.get_ele_type(tool, tool_name)
        for route in self._ele_setters:
            if route.matches(
                ele_type, control_name, tool_name, control_attrib, tool_attrib
            ):
                route.func(
                    tool,
                    value,
                    control_name,
                    tool_name,
                    control_attrib,
                    tool_attrib,
                    **kwargs,
                )
                return
        result = self.default_ele_setter(
            tool, value, control_name, tool_name, control_attrib, tool_attrib
        )
        if result is NotImplemented:
            raise KeyError(
                f"No element setter matched for control_name={control_name!r}, "
                f"control_attrib={control_attrib!r}, ele_type={ele_type!r}"
            )

    def check_ele_routes(
        self,
        tool: Any,
        mappings: list,
    ) -> None:
        """Pre-flight check that element variable mappings route to the correct handlers.

        Temporarily replaces all element getter/setter route functions and the default
        fallbacks with mocks that record the ``control_name``, ``tool_name``,
        ``control_attrib``, and ``tool_attrib`` they receive, keyed by the full
        variable name.  Then calls ``get_property`` and ``set_property`` for every
        mapping's variable name and compares the recorded arguments against the
        expected values on each mapping.

        Raises ``ValueError`` listing every mismatch if any are found.

        Parameters
        ----------
        tool :
            The tool object passed through to route handlers (needed so that
            ``get_ele_type`` can resolve element types for route matching).
        mappings :
            List of ``EleVariableMapping`` objects to verify.  Each must expose
            ``control_name``, ``tool_name``, ``control_attrib``, ``tool_attrib``,
            and ``var.name``.
        """
        received_get: dict[str, dict] = {}
        received_set: dict[str, dict] = {}
        current_var: list[str] = [None]  # mutable cell updated before each call

        def mock_getter(
            t: Any,
            control_name: str,
            tool_name: str,
            control_attrib: str,
            tool_attrib: str,
            **kwargs,
        ) -> None:
            received_get[current_var[0]] = dict(
                control_name=control_name,
                tool_name=tool_name,
                control_attrib=control_attrib,
                tool_attrib=tool_attrib,
            )
            return None

        def mock_setter(
            t: Any,
            value: Any,
            control_name: str,
            tool_name: str,
            control_attrib: str,
            tool_attrib: str,
            **kwargs,
        ) -> None:
            received_set[current_var[0]] = dict(
                control_name=control_name,
                tool_name=tool_name,
                control_attrib=control_attrib,
                tool_attrib=tool_attrib,
            )

        old_getters = self._ele_getters
        old_setters = self._ele_setters
        old_dget = self.default_ele_getter
        old_dset = self.default_ele_setter
        try:
            self.default_ele_getter = mock_getter  # type: ignore[assignment]
            self.default_ele_setter = mock_setter  # type: ignore[assignment]
            self._ele_getters = [
                dataclasses.replace(r, func=mock_getter) for r in old_getters
            ]
            self._ele_setters = [
                dataclasses.replace(r, func=mock_setter) for r in old_setters
            ]

            for mapping in mappings:
                current_var[0] = mapping.var.name
                self.get_property(tool, mapping.var.name)
                self.set_property(tool, mapping.var.name, None)

            mismatches: list[str] = []
            for mapping in mappings:
                expected = dict(
                    control_name=mapping.control_name,
                    tool_name=mapping.tool_name,
                    control_attrib=mapping.control_attrib,
                    tool_attrib=mapping.tool_attrib,
                )
                for label, log in (("get", received_get), ("set", received_set)):
                    received = log.get(mapping.var.name)
                    if received is None:
                        mismatches.append(
                            f"[{label}] {mapping.var.name!r}: no handler was called"
                        )
                    elif received != expected:
                        mismatches.append(
                            f"[{label}] {mapping.var.name!r}: "
                            f"expected {expected!r}, got {received!r}"
                        )

            if mismatches:
                raise ValueError(
                    "Element route pre-flight check failed:\n"
                    + "\n".join(f"  {m}" for m in mismatches)
                )
        finally:
            self._ele_getters = old_getters
            self._ele_setters = old_setters
            self.default_ele_getter = old_dget  # type: ignore[assignment]
            self.default_ele_setter = old_dset  # type: ignore[assignment]
