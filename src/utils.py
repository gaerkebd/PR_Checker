"""
utils.py
--------
Shared helpers: config loading, logging setup.
"""

import logging
import os
import sys

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def require_env(var: str) -> str:
    """Return env var value or raise a clear error."""
    print("checking github env. var")
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            f"Add it to your .env file or export it in your shell."
        )
    return val
