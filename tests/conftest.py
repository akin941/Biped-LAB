import pytest

from bipedlab import upstream


@pytest.fixture(scope="session")
def microduck_rl_path():
    try:
        return upstream.require("microduck_rl")
    except upstream.UpstreamError as exc:
        pytest.skip(str(exc))
