import logging
import pathlib

import matplotlib.pyplot as plt
import pytest

from ...z import ImpactZOutput

z_tests = pathlib.Path(__file__).resolve().parent


test_artifacts = z_tests / "artifacts"

z_examples_root = z_tests / "examples"
z_example1 = z_examples_root / "example1.in"
z_example2 = z_examples_root / "example2.in"
z_example3 = z_examples_root / "example3.in"

z_examples = [z_example1, z_example2, z_example3]
bmad_files = z_tests / "bmad"

logging.getLogger("pytao.subproc").setLevel("WARNING")
logging.getLogger("matplotlib.font_manager").setLevel("WARNING")


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = 0

    def savefig():
        nonlocal index
        filename = test_artifacts / f"{request.node.name}_{index}.png"
        test_artifacts.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
        plt.savefig(filename)
        index += 1

    monkeypatch.setattr(plt, "show", savefig)

    def plot_and_savefig(*args, **kwargs):
        res = orig_plot(*args, **kwargs)
        savefig()
        return res

    orig_plot = ImpactZOutput.plot
    monkeypatch.setattr(ImpactZOutput, "plot", plot_and_savefig)
