import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from . import transformer
from . import bpe
from . import training

__all__ = ["transformer","bpe","training"]
