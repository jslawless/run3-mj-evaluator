from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("run3-mj-evaluator")
except PackageNotFoundError:
    __version__ = "unknown"
