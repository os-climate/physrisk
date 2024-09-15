import pytest
import json
import logging
import os
import pathlib
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
# from deepdiff import DeepDiff

from physrisk.hazard_models.hazard_cache import (
    H3BasedCache,
    LMDBStore,
    MemoryStore,
    Store,
    to_json,
)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@pytest.fixture()
def log_to_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename="test.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@pytest.fixture()
def load_credentials():
    from dotenv import load_dotenv

    load_dotenv("credentials.env")
    # expected by physrisk: to remove when fixed
    if "OSC_S3_BUCKET" not in os.environ:
        os.environ["OSC_S3_BUCKET"] = ""
    if "OSC_S3_ACCESS_KEY" not in os.environ:
        os.environ["OSC_S3_ACCESS_KEY"] = ""
    if "OSC_S3_SECRET_KEY" not in os.environ:
        os.environ["OSC_S3_SECRET_KEY"] = ""
    return "loaded"


@pytest.fixture(autouse=True)
def skip_if_needs_live_data(request, load_credentials):
    env = os.environ.get("PHYSRISK_ENVIRONMENT", "")
    if request.node.get_closest_marker("live_data"):
        if request.node.get_closest_marker("live_data").args[0] != env:
            pytest.skip(f"skipped: PHYSRISK_ENVIRONMENT set to {env}")


@pytest.fixture
def live_working_dir():
    working = Path(__file__).parent.parent / "working"
    working.mkdir(exist_ok=True)
    return working.absolute()


@pytest.fixture
def live_hazard_dir(live_working_dir):
    hazard = Path(live_working_dir) / "hazard"
    hazard.mkdir(exist_ok=True)
    return hazard.absolute()


@pytest.fixture(scope="session")
def working_dir():
    """Location of the working directory to be used for unit and integration tests.

    Returns:
        str: Directory path.
    """
    working = Path(__file__).parent / "working"
    working.mkdir(exist_ok=True)
    return working.absolute()


@pytest.fixture(scope="session")
def hazard_dir():
    """Location of the hazard cache used for unit and integration tests.

    Returns:
        str: Directory path.
    """
    hazard = Path(__file__).parent / "working" / "hazard"
    hazard.mkdir(exist_ok=True, parents=True)
    return hazard.absolute()


@pytest.fixture(scope="session")
def vulnerability_onboarding_dir():
    """Location of the vulnerability on-boarding directory..

    Returns:
        str: Directory path.
    """
    path = Path(__file__).parents[1] / "docs" / "vulnerability_functions"
    return path


# def assert_expected(result: str, func_name: str, update_expected: bool):
#     result_dict, expected_dict = get_result_expected(result, func_name, update_expected)
#     diffs = DeepDiff(expected_dict, result_dict)
#     assert len(diffs.affected_paths) == 0


def get_result_expected(result: str, func_name: str, update_expected: bool):
    path = (
        pathlib.Path(__file__).parent
        / "test_data"
        / "expected"
        / (func_name.replace(".", "-") + ".json")
    )
    if update_expected:
        with open(path, "w") as f:
            f.write(result)
    with open(path, "r") as f:
        expected_dict = json.loads(f.read())
    result_dict = json.loads(result)
    return result_dict, expected_dict


@contextmanager
def cache_store_tests(func_name: str, update: bool = True):
    base_path = pathlib.Path(__file__).parent / "test_data" / "inputs"
    file_root = func_name.replace(".", "#")
    # we have human-readable data in the JSON cache. This goes into source control.
    file = str(base_path / file_root) + "#hazard_cache.json"
    # this is the temporary database file
    temp_lmdb_file = str(base_path / file_root) + "#hazard_cache_temp.db"
    cache_store = load_inputs_cache_store(file)
    try:
        # we unpack the human-readable JSON into the (non-human-readable) LMDB cache
        temp_cache = H3BasedCache(LMDBStore(temp_lmdb_file))
        # stored as strings in LMDB, hence dumps
        temp_cache.setitems({k: json.dumps(v) for k, v in cache_store.getall().items()})
        yield temp_cache
    finally:
        # take contents of LMDB and save back as human-readbale JSON
        if update:
            save_inputs_cache_store(temp_cache.store, file)
        # delete LMDB files: this should not be persisted in source control
        shutil.rmtree(temp_lmdb_file)


def load_inputs_cache_store(file: str):
    # by default, use LMDB to cache requests made via API
    return MemoryStore(file=file)


def save_inputs_cache_store(store: Store, file: str):
    with open(file, "w") as f:
        f.write(to_json(store))


def pytest_addoption(parser):
    parser.addoption(
        "--update_expected",
        action="store",
        help="Update the test expected result if true",
    )
    parser.addoption(
        "--update_inputs", action="store", help="Update the test mocked data if true"
    )


@pytest.fixture
def update_expected(request):
    return request.config.getoption("--update_expected")


@pytest.fixture
def update_inputs(request):
    return request.config.getoption("--update_inputs")
