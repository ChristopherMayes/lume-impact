from impact import Impact
import impact
from pathlib import Path

root = Path(impact.__file__).parent


def test_tesla_9cell_cavity():
    input_file = root / "tests/input/tesla_9cell_cavity/ImpactT.in"
    assert input_file.exists()

    I = Impact(input_file)
    I.header["Np"] = 1
    I.run()

    assert I.stat("mean_kinetic_energy")[-1] == 26900553.0
