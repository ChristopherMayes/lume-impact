import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))
