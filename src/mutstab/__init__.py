"""mutstab — predicting mutation effects on protein melting temperature (Tm)."""

__version__ = "0.1.0"

from mutstab import paths
from mutstab.config import Config, load_config

__all__ = ["Config", "load_config", "paths", "__version__"]
