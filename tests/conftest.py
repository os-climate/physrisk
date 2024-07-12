import pytest


def pytest_addoption(parser):
    parser.addoption("--update_expected", action="store", help="Update the test expected result if true")
    parser.addoption("--update_inputs", action="store", help="Update the test mocked data if true")


@pytest.fixture
def update_expected(request):
    return request.config.getoption("--update_expected")


@pytest.fixture
def update_inputs(request):
    return request.config.getoption("--update_inputs")
