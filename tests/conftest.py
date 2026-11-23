import pytest
from synthetic import write_m5_files


@pytest.fixture(scope="session")
def m5_dir(tmp_path_factory):
    return write_m5_files(tmp_path_factory.mktemp("m5raw"))


@pytest.fixture(scope="session")
def m5data(m5_dir):
    from m5.data import load_m5

    return load_m5(m5_dir)
