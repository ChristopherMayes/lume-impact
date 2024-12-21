import pytest

from impact.z.input import ImpactZInput
from ...z import ImpactZ
from .conftest import z_example1


examples = pytest.mark.parametrize(
    ("example_input_file",),
    [
        pytest.param(z_example1, id="example1"),
    ],
)


@examples
def test_load_from_filename(example_input_file: ImpactZInput) -> None:
    ImpactZ(input=example_input_file)


@examples
def test_load_from_contents(example_input_file: ImpactZInput) -> None:
    with open(example_input_file, "rt") as fp:
        contents = fp.read()

    ImpactZ(input=contents)


@examples
def test_load_from_instance(example_input_file: ImpactZInput) -> None:
    input = ImpactZInput.from_file(example_input_file)
    ImpactZ(input=input)


@examples
def test_run_example(example_input_file: ImpactZInput) -> None:
    I = ImpactZ(input=example_input_file)
    output = I.run()
    print(output)
