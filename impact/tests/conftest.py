import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session", autouse=True)
def lcls_archive_file(rootdir):
    return f"{rootdir}/files/lcls_injector.h5"

