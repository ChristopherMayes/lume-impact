from __future__ import annotations

import ast
import math
import re
from collections.abc import Sequence

import numpy as np

from .types import BaseModel


class InputLine(BaseModel):
    header_comments: list[str] = []
    inline_comment: str | None = None
    data: Sequence[float | int]


re_missing_exponent = re.compile(r"([+-]?\d*\.\d+)([+-]\d+)")


def fix_line(contents: str) -> str:
    contents = contents.replace("D", "E").replace("d", "e")  # fortran float style
    return re_missing_exponent.sub(r"\1E\2", contents)


def parse_input_line(line: str) -> InputLine:
    if "/" in line:
        line, comment = line.split("/", 1)
        comment = comment.strip()
    else:
        comment = None

    line = line.replace("D", "E").replace("d", "e")  # fortran float style
    line = re_missing_exponent.sub(r"\1E\2", line)

    parts = line.strip().split()

    def literal_eval(value: str):
        if value.lower() == "nan":
            return math.nan
        if value.lower().startswith("-inf"):
            return -math.inf
        if value.lower().startswith("inf"):
            return math.inf
        try:
            return ast.literal_eval(value)
        except Exception:
            raise ValueError(
                f"Input line: {line!r} failed to parse element {value!r} "
                f"as it does not correspond to a valid Python constant."
            ) from None

    return InputLine(
        data=[literal_eval(value) for value in parts],
        inline_comment=comment,
        header_comments=[],
    )


def parse_input_lines(lines: str | Sequence[str]) -> list[InputLine]:
    if isinstance(lines, str):
        lines = lines.splitlines()

    input_lines = []
    comments = []
    for line in lines:
        line = line.lstrip()
        if line.startswith("!"):
            comments.append(line.lstrip("! "))
        else:
            input_line = parse_input_line(line)
            if input_line.data:
                input_lines.append(input_line)
                input_line.header_comments = comments
                comments = []

    return input_lines


def read_input_file(filename):
    with open(filename) as fp:
        return parse_input_lines(fp.read().splitlines())


def lines_to_ndarray(lines: Sequence[InputLine]) -> np.ndarray:
    return np.asarray([line.data for line in lines])
