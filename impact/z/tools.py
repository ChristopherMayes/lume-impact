from __future__ import annotations

import datetime
import enum
import functools
import html
import importlib
import inspect
import logging
import pathlib
import string
import subprocess
import sys
import traceback
import uuid
from typing import Any, Union
from collections.abc import Mapping, Sequence

import prettytable
import pydantic
import pydantic_settings
from ..repr import rich_format

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from types import UnionType
except ImportError:
    # Python < 3.10
    union_types = {Union}
else:
    union_types = {UnionType, Union}


logger = logging.getLogger(__name__)


class DisplayOptions(
    pydantic_settings.BaseSettings,
    env_prefix="LUME_",
    case_sensitive=False,
):
    """
    jupyter_render_mode : One of {"html", "markdown", "native", "repr"}
        Defaults to "repr".
        Environment variable: LUME_JUPYTER_RENDER_MODE.
    console_render_mode : One of {"markdown", "native", "repr"}
        Defaults to "repr".
        Environment variable: LUME_CONSOLE_RENDER_MODE.
    include_description : bool, default=True
        Include descriptions in table representations.
        Environment variable: LUME_INCLUDE_DESCRIPTION.
    ascii_table_type : int, default=prettytable.MARKDOWN
        Default to a PrettyTable markdown ASCII table.
        Environment variable: LUME_ASCII_TABLE_TYPE.
    filter_tab_completion : bool, default=True
        Filter out unimportant details (pydantic methods and such) from classes.
        Environment variable: LUME_FILTER_TAB_COMPLETION.
    verbose : int, default=0
        At level 0, hide output during `run()` by default.
        At level 1, show output during `run()` by default.
        Equivalent to configuring the default setting of `ImpactZ.verbose` to `True`.
        Environment variable: LUME_VERBOSE.
    """

    # TODO: change 'genesis' to 'native' in lume-genesis
    jupyter_render_mode: Literal["html", "markdown", "native", "repr"] = "repr"
    console_render_mode: Literal["markdown", "native", "repr"] = "repr"
    include_description: bool = True
    ascii_table_type: int = prettytable.TableStyle.MARKDOWN
    verbose: int = 0
    filter_tab_completion: bool = True


global_display_options = DisplayOptions()


def execute(cmd, cwd=None):
    """
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")

    Useful in Jupyter notebook

    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    assert popen.stdout is not None
    yield from iter(popen.stdout.readline, "")
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# Alternative execute
def execute2(cmd, timeout=None, cwd=None, encoding="utf-8"):
    """
    Execute with time limit (timeout) in seconds, catching run errors.
    """
    output = {"error": True, "log": ""}
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output["log"] = p.stdout
        output["error"] = False
        output["why_error"] = ""
    except subprocess.TimeoutExpired as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join(
            (
                stdout.decode(encoding, errors="ignore"),
                f"{ex.__class__.__name__}: {ex}",
            )
        )
        output["why_error"] = "timeout"
    except subprocess.CalledProcessError as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join(
            (
                stdout.decode(encoding, errors="ignore"),
                f"{ex.__class__.__name__}: {ex}",
            )
        )
        output["why_error"] = "error"
    except Exception as ex:
        stack = traceback.print_exc()
        output["log"] = f"Unknown run error: {ex.__class__.__name__}: {ex}\n{stack}"
        output["why_error"] = "unknown"
    return output


def isotime():
    """UTC to ISO 8601 with Local TimeZone information without microsecond"""
    return (
        datetime.datetime.now(datetime.UTC)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"


@functools.cache
def get_output_mode() -> OutputMode:
    """
    Get the output mode for lume-impact objects.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    OutputMode
        The detected output mode.
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return OutputMode.plain

    from IPython.display import display

    class ReprCheck:
        mode: OutputMode = OutputMode.unknown

        def _repr_html_(self) -> str:
            self.mode = OutputMode.html
            return (
                "<!-- lume-impact detected Jupyter and will use HTML for rendering. -->"
            )

        def __repr__(self) -> str:
            self.mode = OutputMode.plain
            return ""

    check = ReprCheck()
    display(check)
    return check.mode


def is_jupyter() -> bool:
    """Is Jupyter detected?"""
    return get_output_mode() == OutputMode.html


def import_by_name(clsname: str) -> type:
    """
    Import the given class or function by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pcdsdevices.device_types.IPM"``

    Returns
    -------
    type
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")


def _truncated_string(value, max_length: int) -> str:
    """
    Truncate a string representation of ``value`` if it's longer than
    ``max_length``.

    Parameters
    ----------
    value :
    max_length : int

    Returns
    -------
    str
    """
    value = str(value)
    if len(value) < max_length + 3:
        return value
    value = value[:max_length]
    return f"{value}..."


def _clean_annotation(annotation) -> str:
    """Clean an annotation for showing to the user."""
    if inspect.isclass(annotation):
        return annotation.__name__
    annotation = str(annotation)
    for remove in [
        "typing.",
        "typing_extensions.",
    ]:
        annotation = annotation.replace(remove, "")

    if annotation.startswith("Literal['"):
        # This is a bit of an implementation detail we don't necessarily need
        # to expose to the user; Literal['type'] is used to differentiate
        # beamline elements and namelists during deserialization.
        return "str"
    return annotation


def table_output(
    obj: pydantic.BaseModel | dict[str, Any],
    display_options: DisplayOptions = global_display_options,
    descriptions: Mapping[str, str | None] | None = None,
    annotations: Mapping[str, str | None] | None = None,
    headers: Sequence[str] | None = None,
):
    """
    Create a table based on user settings for the given object.

    In Jupyter (with "html" render mode configured), this will display
    an HTML table.

    In the terminal, this will create a markdown ASCII table.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.
    """
    if is_jupyter() and display_options.jupyter_render_mode != "markdown":

        class _InfoObj:
            def _repr_html_(_self) -> str:
                return html_table_repr(
                    obj,
                    seen=[],
                    descriptions=descriptions,
                    annotations=annotations,
                    display_options=display_options,
                    headers=headers,
                )

        return _InfoObj()

    ascii_table = ascii_table_repr(
        obj,
        seen=[],
        display_options=display_options,
        descriptions=descriptions,
        annotations=annotations,
        headers=headers,
    )
    print(ascii_table)


def _copy_to_clipboard_html(contents: str) -> str:
    """Create copy-to-clipboard HTML for the given text."""
    return string.Template(
        """
        <div style="display: flex; justify-content: flex-end;">
          <button class="copy-${hash_}">
            Copy to clipboard
          </button>
          <br />
        </div>
        <script type="text/javascript">
          function copy_to_clipboard(text) {
            navigator.clipboard.writeText(text).then(
              function () {
                console.log("Copied to clipboard:", text);
              },
              function (err) {
                console.error("Failed to copy to clipboard:", err, text);
              },
            );
          }
          var copy_button = document.querySelector(".copy-${hash_}");
          copy_button.addEventListener("click", function (event) {
            copy_to_clipboard(`${table}`);
          });
        </script>
        """
    ).substitute(
        hash_=uuid.uuid4().hex,
        table=contents.replace("`", r"\`"),
    )


def _get_table_fields(
    obj: pydantic.BaseModel | dict[str, Any],
    descriptions: Mapping[str, str | None] | None = None,
    annotations: Mapping[str, str | None] | None = None,
):
    """Get values, descriptions, and annotations for a table."""
    if isinstance(obj, pydantic.BaseModel):
        fields = {
            attr: getattr(obj, attr, None)
            for attr, field_info in obj.model_fields.items()
            if field_info.repr
        }
        if annotations is None:
            annotations = {
                attr: field_info.annotation
                for attr, field_info in obj.model_fields.items()
            }
        if descriptions is None:
            descriptions = {
                attr: field_info.description
                for attr, field_info in obj.model_fields.items()
            }
    else:
        fields = obj
        if annotations is None:
            annotations = {attr: "" for attr in fields}
        if descriptions is None:
            descriptions = {attr: "" for attr in fields}

    return fields, descriptions, annotations


def html_table_repr(
    obj: pydantic.BaseModel | dict[str, Any],
    seen: list,
    display_options: DisplayOptions = global_display_options,
    descriptions: Mapping[str, str | None] | None = None,
    annotations: Mapping[str, str | None] | None = None,
    headers: Sequence[str] | None = None,
) -> str:
    """
    Pydantic model table HTML representation for Jupyter.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.

    Returns
    -------
    str
        HTML table representation.
    """
    # TODO: generalize these tables; callers have a confusing mapping if they
    # change the headers
    headers = headers or ["Attribute", "Value", "Type", "Description"]
    assert len(headers) == 4

    include_description = display_options.include_description and headers[-1]

    seen.append(id(obj))
    rows = []
    fields, descriptions, annotations = _get_table_fields(
        obj, descriptions, annotations
    )

    for attr, value in fields.items():
        if value is None:
            continue
        annotation = annotations.get(attr, "")
        description = descriptions.get(attr, "")

        if isinstance(value, (pydantic.BaseModel, dict)):
            if id(value) in seen:
                table_value = "(recursed)"
            else:
                table_value = html_table_repr(
                    value,
                    seen,
                    display_options=display_options,
                    headers=headers,
                )
        else:
            table_value = html.escape(_truncated_string(value, max_length=100))

        annotation = html.escape(_clean_annotation(annotation))
        if display_options.include_description:
            description = html.escape(description or "")
            description = f'<td style="text-align: left;">{description}</td>'
        else:
            description = ""

        rows.append(
            f"<tr>"
            f"<td>{attr}</td>"
            f"<td>{table_value}</td>"
            f"<td>{annotation}</td>"
            f"{description}"
            f"</tr>"
        )

    raw = rich_format(obj)
    copy_to_clipboard = _copy_to_clipboard_html(raw)

    return "\n".join(
        [
            copy_to_clipboard,
            '<table style="table td:nth-child(3) { text-align: start; }">',
            " <tr>",
            f"  <th>{headers[0]}</th>",
            f"  <th>{headers[1]}</th>",
            f"  <th>{headers[2]}</th>",
            f"  <th>{headers[3]}</th>" if include_description else "",
            " </tr>",
            "</th>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
        ]
    )


def ascii_table_repr(
    obj: pydantic.BaseModel | dict[str, Any],
    seen: list,
    display_options: DisplayOptions = global_display_options,
    descriptions: Mapping[str, str | None] | None = None,
    annotations: Mapping[str, str | None] | None = None,
    headers: Sequence[str] | None = None,
) -> prettytable.PrettyTable:
    """
    Pydantic model table ASCII representation for the terminal.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.

    Returns
    -------
    str
        HTML table representation.
    """
    headers = headers or ["Attribute", "Value", "Type", "Description"]
    assert len(headers) == 4

    seen.append(id(obj))
    rows = []
    fields, descriptions, annotations = _get_table_fields(
        obj, descriptions, annotations
    )
    annotations = annotations or {}

    for attr, value in fields.items():
        if value is None:
            continue
        description = descriptions.get(attr, "")
        annotation = annotations.get(attr, "")

        if isinstance(value, pydantic.BaseModel):
            if id(value) in seen:
                table_value = "(recursed)"
            else:
                table_value = str(
                    ascii_table_repr(value, seen, display_options=display_options)
                )
        else:
            table_value = _truncated_string(value, max_length=30)

        rows.append(
            (
                attr,
                table_value,
                _clean_annotation(annotation),
                description or "",
            )
        )

    headers = list(headers)
    if not display_options.include_description or not headers[-1]:
        headers = headers[:3]
        # Chop off the description for each row
        rows = [row[:-1] for row in rows]

    table = prettytable.PrettyTable(field_names=headers)
    table.add_rows(rows)
    table.set_style(display_options.ascii_table_type)
    return table


def check_if_existing_path(input: str) -> pathlib.Path | None:
    """
    Check if the ``input`` path exists, and convert it to a `pathlib.Path`.

    Parameters
    ----------
    input : str

    Returns
    -------
    pathlib.Path or None
    """
    path = pathlib.Path(input).resolve()
    try:
        if path.exists():
            return path
    except OSError:
        ...
    return None


def read_if_path(
    input: pathlib.Path | str,
    source_path: pathlib.Path | str | None = None,
) -> tuple[pathlib.Path | None, str]:
    """
    Read ``input`` if it's an existing path.

    Parameters
    ----------
    input : pathlib.Path or str
        Filename *or* source contents.
    source_path : pathlib.Path or str, optional
        The path where ``input`` may be relative to.

    Returns
    -------
    Optional[pathlib.Path]
        The filename, if it was read out.
    str
        If ``input`` was an existing file, this is the string contents of that
        file.
        Otherwise, it is the source string ``input``.
    """
    if not input:
        return None, input

    if source_path is None:
        source_path = pathlib.Path(".")

    source_path = pathlib.Path(source_path)

    if isinstance(input, pathlib.Path):
        path = input
    else:
        path = check_if_existing_path(str(source_path / input))
        if not path:
            return None, str(input)

    # Update our source path; we found a file.  This is probably what
    # the user wants.
    with open(path) as fp:
        return path.absolute(), fp.read()
