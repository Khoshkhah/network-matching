"""
Lightweight logging for network_matching.

The package logs under the ``network_matching`` logger (with per-module children like
``network_matching.matcher``). By default nothing is emitted (a ``NullHandler`` is attached in
``__init__``); call :func:`setup_logging` to write to a timestamped file under ``logs/`` (and
optionally the console).

Example
-------
>>> from network_matching import setup_logging, DuckDBMapMatcher
>>> setup_logging()                       # logs/network_matching_YYYYmmdd_HHMMSS.log
>>> m = DuckDBMapMatcher(); ...
>>> m.match_routes()                       # progress + timing go to the log file
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

ROOT_LOGGER_NAME = "network_matching"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the package logger (or a child ``network_matching.<name>``)."""
    if not name:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def setup_logging(
    logfile: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    log_dir: str = "logs",
) -> Tuple[logging.Logger, str]:
    """Configure file (and optional console) logging for the package.

    Parameters
    ----------
    logfile:
        Explicit log file path. If ``None``, a timestamped file
        ``<log_dir>/network_matching_YYYYmmdd_HHMMSS.log`` is created.
    level:
        Logging level (default ``logging.INFO``).
    console:
        Also echo to stdout.
    log_dir:
        Directory for the auto-named log file (created if missing).

    Returns
    -------
    (logger, logfile_path)
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Drop any handlers from a previous setup_logging call (idempotent).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if logfile is None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f"{ROOT_LOGGER_NAME}_{ts}.log")
    else:
        parent = os.path.dirname(logfile)
        if parent:
            os.makedirs(parent, exist_ok=True)

    fmt = logging.Formatter(_DEFAULT_FORMAT)

    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(level)
        logger.addHandler(ch)

    logger.propagate = False
    logger.info("Logging initialized -> %s", logfile)
    return logger, logfile
