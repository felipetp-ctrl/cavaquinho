from importlib.metadata import PackageNotFoundError, version

from .core import Caco, Validator
from .models import ClaimResult, Labels, ValidationResult

try:
    __version__ = version("cavaquinho")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "Validator",
    "Caco",
    "ClaimResult",
    "ValidationResult",
    "Labels",
    "__version__",
]


def __getattr__(name: str):
    if name == "caco":
        import warnings
        warnings.warn(
            "'caco' is deprecated and will be removed in v1.0. Use 'Caco' or 'Validator'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Validator
    raise AttributeError(f"module 'cavaquinho' has no attribute {name!r}")
