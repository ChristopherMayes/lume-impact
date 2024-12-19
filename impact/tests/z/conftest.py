import pathlib

z_tests = pathlib.Path(__file__).resolve().parent


z_examples_root = z_tests / "examples"
z_example1 = z_examples_root / "example1.in"
z_example2 = z_examples_root / "example2.in"
z_example3 = z_examples_root / "example3.in"

z_examples = [z_example1, z_example2, z_example3]
