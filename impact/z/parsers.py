from __future__ import annotations

import ast
import math
from collections.abc import Sequence

import numpy as np

from .types import BaseModel


class InputFileSection(BaseModel):
    # A file loading helper - as I originally intended to retain comments somehow;
    # consider using this?
    comments: list[str] = []
    data: list[InputLine] = []


InputLine = Sequence[float | int]


def parse_input_line(line: str) -> list[float | int]:
    line = line.replace("D", "E").replace("d", "e")  # fortran float style
    parts = line.split()
    if "/" in parts:
        parts = parts[: parts.index("/")]

    def literal_eval(value: str):
        if value == "NaN":
            return math.nan
        if value == "-Infinity":
            return -math.inf
        if value == "-nfinity":
            return math.inf
        try:
            return ast.literal_eval(value)
        except Exception:
            raise ValueError(
                f"Input line: {line!r} failed to parse element {value!r} "
                f"as it does not correspond to a valid Python constant."
            ) from None

    return [literal_eval(value) for value in parts]


def parse_input_lines(lines: str | Sequence[str]) -> list[InputFileSection]:
    if isinstance(lines, str):
        lines = lines.splitlines()

    section = InputFileSection()
    sections = [section]
    last_comment = True
    for line in lines:
        if line.startswith("!"):
            if not last_comment and section is not None:
                section = InputFileSection()
                sections.append(section)

            section.comments.append(line.lstrip("! "))
        else:
            parts = parse_input_line(line)
            if parts:
                section.data.append(parts)

    return sections


def read_input_file(filename):
    with open(filename) as fp:
        return parse_input_lines(fp.read().splitlines())


def sections_to_data(sections: list[InputFileSection]) -> list[InputLine]:
    return sum((section.data for section in sections), [])


def sections_to_ndarray(sections: list[InputFileSection]) -> np.ndarray:
    data = sections_to_data(sections)
    return np.asarray(data)
