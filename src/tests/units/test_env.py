import zoneinfo

import app.env


def test_env_vars() -> None:
    """Test whether the environment variables are read correctly.

    Expected behavior:
        - Values being identical to the ones in the pyproject.toml file.
    """

    assert app.env.STREAMING_LOG_LEVEL == 30
    assert app.env.FILE_LOG_LEVEL == 10
    assert app.env.LOG_FMT == "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
    assert app.env.LOG_DATEFMT == "%Y-%m-%dT%H:%M:%SZ"
    assert app.env.LOG_FILEPATH == "./test_report/runtime.log"
    assert app.env.TZ == zoneinfo.ZoneInfo("UTC")
