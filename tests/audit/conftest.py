import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Register marker dynamically in audit subtree so pyproject.toml remains pristine and read-only.
    config.addinivalue_line(
        "markers",
        "audit: adversarial production-readiness audit tests (defect reproductions, oracles, boundary suites)",
    )
