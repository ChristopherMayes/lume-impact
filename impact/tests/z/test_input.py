import logging
import pathlib
import re

import numpy as np
import pytest

from ...z.input import ImpactZInput, parse_input_line


logger = logging.getLogger(__name__)
z_tests = pathlib.Path(__file__).resolve().parent
impact_z_examples = z_tests / "examples"


def normalize_source(source: str) -> str:
    """
    Normalize input file source for comparison.

    Removes comments, normalizes whitespace to a single space, and fixes
    floating point representation.
    """
    lines = source.strip().splitlines()

    def fix_line(line: str) -> str:
        line = line.replace("d", "e")
        line = line.replace("D", "e")

        line = line.split("/", 1)[0]

        # Any whitespace becomes a single string:
        return re.sub(r"\s+", " ", line).strip()

    return "\n".join(
        fix_line(line) for line in lines if not line.startswith("!") and line.strip()
    )


def compare_inputs(expected: str, generated: str):
    expected = normalize_source(expected)
    generated = normalize_source(generated)

    expected_lines = expected.splitlines()
    generated_lines = generated.splitlines()
    for lineno, (line_expected, line_generated) in enumerate(
        zip(expected_lines, generated_lines),
        start=1,
    ):
        expected_parts = parse_input_line(line_expected)
        generated_parts = parse_input_line(line_generated)

        if lineno == 1 and len(expected_parts) != len(generated_parts):
            # NOTE: a bit of first line handling for GPU settings
            # this is a "do not care about GPU setting" section
            if len(expected_parts) == 3 and len(generated_parts) == 2:
                generated_parts.append(expected_parts[-1])
            elif len(expected_parts) == 2 and len(generated_parts) == 3:
                expected_parts.append(generated_parts[-1])

        if len(expected_parts) != len(generated_parts):
            if len(generated_parts) > len(expected_parts):
                num_extra = len(generated_parts) - len(expected_parts)
                if np.allclose(generated_parts[-num_extra:], 0):
                    msg = f"Ignoring extra input generated on line {line_generated} (expected: {line_expected})"
                    # TODO: we may circle back to this and make a different decision
                    print(msg)
                    logger.warning(msg)
                    continue

            raise ValueError(
                f"Generated input file differs on line {lineno}.\n"
                f"Expected line:  {line_expected!r}\n"
                f"Generated line: {line_generated!r}\n"
                f"Differs in number of parts {len(expected_parts)} != {len(generated_parts)}"
            )

        assert np.allclose(generated_parts, expected_parts)

    if len(expected_lines) != len(generated_lines):
        raise ValueError(
            f"Expected line count {len(expected_lines)} differs "
            f"vs generated lines {len(generated_lines)}"
        )


@pytest.mark.parametrize(
    ("filename",),
    [pytest.param(fn, id=fn.name) for fn in impact_z_examples.glob("*.in")],
)
def test_load_input_smoke(filename: pathlib.Path) -> None:
    ImpactZInput.from_file(filename)


@pytest.mark.parametrize(
    ("filename",),
    [pytest.param(fn, id=fn.name) for fn in impact_z_examples.glob("*.in")],
)
def test_input_roundtrip(filename: pathlib.Path) -> None:
    loaded = ImpactZInput.from_file(filename)
    with open(filename, "rt") as fp:
        on_disk_contents = fp.read()

    serialized_contents = loaded.to_contents()
    compare_inputs(on_disk_contents, serialized_contents)
