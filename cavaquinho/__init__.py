from importlib.metadata import PackageNotFoundError, version
from .core import Validator, caco
from .models import ClaimResult, ValidationResult, Labels

try:
    __version__ = version("cavaquinho")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Validator",
    "caco",
    "ClaimResult",
    "ValidationResult",
    "Labels",
    "__version__",
]
