from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("smoldynutils")
except PackageNotFoundError:
    # package not installed (e.g. local dev)
    __version__ = "0.0.0"
