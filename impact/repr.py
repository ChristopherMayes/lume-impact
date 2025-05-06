import dataclasses
import io
import logging

import numpy as np
import pydantic
import rich.console
import rich.terminal_theme
from rich.repr import RichReprResult

logger = logging.getLogger(__name__)
style_hotfix = (
    '''style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"''',
    '''style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;margin-left: 0px;margin-top: 0.5em;"''',
)


def is_default(value, default):
    if isinstance(default, pydantic.fields.FieldInfo):
        if default.default_factory in (dict, list, tuple, set):
            return not bool(value)
        default = default.default

    try:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return default == value
    except Exception:
        return False


def _dataclass_rich_repr_without_defaults(obj) -> RichReprResult:
    """Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects."""

    fields = dataclasses.fields(obj)
    for fld in fields:
        if not fld.repr:
            continue
        value = getattr(obj, fld.name)
        if dataclasses.is_dataclass(value):
            child_repr = list(_dataclass_rich_repr_with_defaults(value))
            yield fld.name, child_repr
        elif not is_default(value, fld.default):
            yield fld.name, value


def _dataclass_rich_repr_with_defaults(obj) -> RichReprResult:
    """Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects."""
    fields = dataclasses.fields(obj)
    for fld in fields:
        if fld.repr:
            value = getattr(obj, fld.name)
            yield fld.name, value


def _pydantic_rich_repr_with_defaults(obj) -> RichReprResult:
    """Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects."""
    fields = obj.model_fields
    for name, field_repr in obj.__repr_args__():
        if name is None:
            yield field_repr
        elif fields[name].repr:
            yield name, field_repr


def _pydantic_rich_repr_without_defaults(obj) -> RichReprResult:
    """Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects."""
    fields = obj.model_fields
    for name, field_repr in obj.__repr_args__():
        if name is None:
            yield field_repr
        else:
            value = getattr(obj, name, None)
            if not is_default(value, fields[name].default):
                yield name, field_repr


_orig_rich_repr = pydantic.BaseModel.__rich_repr__
_rich_terminal_themes = {
    "default": rich.terminal_theme.DEFAULT_TERMINAL_THEME,
    "monokai": rich.terminal_theme.MONOKAI,
    "dimmed_monokai": rich.terminal_theme.DIMMED_MONOKAI,
    "solarized_light": rich.terminal_theme.TerminalTheme(
        (0xFD, 0xF6, 0xE3),
        (0x58, 0x6E, 0x75),
        [
            (0x00, 0x2B, 0x36),
            (0xDC, 0x32, 0x2F),
            (0x85, 0x99, 0x00),
            (0xB5, 0x89, 0x00),
            (0x26, 0x8B, 0xD2),
            (0x6C, 0x71, 0xC4),
            (0x2A, 0xA1, 0x98),
            (0x93, 0xA1, 0xA1),
        ],
        [
            (0x65, 0x7B, 0x83),
            (0xDC, 0x32, 0x2F),
            (0x85, 0x99, 0x00),
            (0xB5, 0x89, 0x00),
            (0x26, 0x8B, 0xD2),
            (0x6C, 0x71, 0xC4),
            (0x2A, 0xA1, 0x98),
            (0xFD, 0xF6, 0xE3),
        ],
    ),
}


def get_rich_terminal_theme(theme: str) -> rich.terminal_theme.TerminalTheme:
    return _rich_terminal_themes[theme]


def _internal_rich_console():
    console = rich.console.Console(file=io.StringIO(), record=True)
    console.is_jupyter = False
    return console


def rich_html_repr(obj, theme: str = "default") -> str:
    console = _internal_rich_console()
    console.print(obj)
    return console.export_html(theme=_rich_terminal_themes[theme])


def rich_format(obj):
    console = _internal_rich_console()
    console.print(obj)
    return console.export_text()


def rich_html_model_repr(
    obj, include_defaults: bool = False, theme: str = "default"
) -> str:
    console = _internal_rich_console()
    try:
        if include_defaults:
            # pytao.TaoStartup.__rich_repr__ = _dataclass_rich_repr_with_defaults
            pydantic.BaseModel.__rich_repr__ = _pydantic_rich_repr_with_defaults
        else:
            pydantic.BaseModel.__rich_repr__ = _pydantic_rich_repr_without_defaults
            # pytao.TaoStartup.__rich_repr__ = _dataclass_rich_repr_without_defaults
        console.print(obj)
        return console.export_html(theme=_rich_terminal_themes[theme]).replace(
            *style_hotfix
        )
    finally:
        pydantic.BaseModel.__rich_repr__ = _orig_rich_repr
        # pytao.TaoStartup.__rich_repr__ = _dataclass_rich_repr_with_defaults


def detailed_html_repr(obj, theme: str = "solarized_light") -> str:
    if not isinstance(obj, pydantic.BaseModel):
        return rich_html_repr(obj)
    without_defaults = rich_html_model_repr(obj, include_defaults=False, theme=theme)
    with_defaults = rich_html_model_repr(obj, include_defaults=True, theme=theme)

    if without_defaults == with_defaults or len(without_defaults) >= 0.9 * len(
        with_defaults
    ):
        return without_defaults

    return f"""
        {without_defaults}
        <details>
            <summary>Including defaults</summary>
            {with_defaults}
        </details>
        """
