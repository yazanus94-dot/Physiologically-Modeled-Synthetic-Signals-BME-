"""Shared environment-version recorder.

Import ``get_versions()`` and stamp it onto experiment metadata
(``metrics.json``, reports) so a real-data result is reproducible from a
known environment.  Centralised so P1/P2/P3 metadata stay consistent.
"""

from typing import Any


def get_versions() -> dict[str, Any]:
    """Return a dict of installed package versions and platform info."""
    def _ver(name):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__",
                           getattr(mod, "version", None))
        except Exception:  # noqa: BLE001
            return None

    import platform
    import sys
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": _ver("numpy"),
        "scipy": _ver("scipy"),
        "pandas": _ver("pandas"),
        "wfdb": _ver("wfdb"),
        "matplotlib": _ver("matplotlib"),
        "scikit_learn": _ver("sklearn"),
        "joblib": _ver("joblib"),
        "seaborn": _ver("seaborn"),
    }

