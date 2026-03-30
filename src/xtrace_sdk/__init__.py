"""XTrace SDK — privacy-preserving encrypted vector and memory SDK."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xtrace-ai-sdk")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "unknown"
