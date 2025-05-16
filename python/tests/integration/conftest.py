import pytest
from pathlib import Path
from dotenv import load_dotenv

# These will store the values determined by pytest_configure and used by fixtures.
_configured_target_env: str = ""
_configured_env_file_path: Path = Path()

def pytest_addoption(parser):
    """Adds a command-line option to pytest for specifying the test environment."""
    parser.addoption(
        "--test-env", 
        action="store", 
        default="prod",  # Default to 'prod' if not specified
        help="Specify the test environment (e.g., dev, prod). Defaults to 'prod'."
    )

def pytest_configure(config):
    """Configures target environment and loads .env file based on --test-env CLI option."""
    global _configured_target_env, _configured_env_file_path
    
    _configured_target_env = config.getoption("test_env").lower()
    # __file__ in conftest.py (in tests/integration/) refers to tests/integration/conftest.py
    _configured_env_file_path = Path(__file__).parent / f".env.{_configured_target_env}"

    if _configured_env_file_path.exists():
        load_dotenv(dotenv_path=_configured_env_file_path, override=True)
        if config.getoption('verbose') > 0:
            print(f"INFO: Loaded environment variables from {_configured_env_file_path} for '{_configured_target_env}' environment.")
    else:
        # This print is for informational purposes if running pytest with -s or -v.
        # The tests will fail later via UUID getters if essential env vars are not loaded.
        print(f"INFO: Environment file {_configured_env_file_path} for '{_configured_target_env}' environment not found. "
              f"Tests may fail if required UUIDs are not set.")

@pytest.fixture(scope="session")
def target_env() -> str:
    """Pytest fixture to provide the configured target environment string."""
    return _configured_target_env

@pytest.fixture(scope="session")
def env_file_path() -> Path:
    """Pytest fixture to provide the Path to the loaded .env file."""
    return _configured_env_file_path
