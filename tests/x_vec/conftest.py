import pytest
from typing import Any

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--key_len",
        action="store",
        default=1024,
        help="key length to pass to test functions",
    )
    parser.addoption(
        "--num_runs",
        action="store",
        default=10,
        help="number of test runs to perform",
    )
    parser.addoption(
        "--alpha_len",
        action="store",
        default=50,
        help="alpha length to pass to test functions",
    )

@pytest.fixture
def key_len(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--key_len"))

@pytest.fixture
def num_runs(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--num_runs"))

@pytest.fixture
def alpha_len(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--alpha_len"))
