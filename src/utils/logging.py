# src/utils/logging.py

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def _ensure_dir(p: PathLike):
    Path(p).mkdir(parents=True, exist_ok=True)


def make_logger(
    name: str = "tdar",
    log_dir: Optional[PathLike] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Create a logger with:
      - Console handler (stdout)
      - Optional file handler in log_dir/log_file

    Notes:
      - We only create 'log_dir' (a subdirectory) if provided.
      - Do NOT pass top-level dirs like "checkpoints/" unless you created it already.

    Args:
      name: logger name
      log_dir: directory to save log file (optional)
      log_file: file name, default: f"{name}.log"
      level: logging level
      console: whether to output to stdout
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs

    # If handlers already exist, just return
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        _ensure_dir(log_dir)
        if log_file is None:
            log_file = f"{name}.log"
        fh = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def make_run_log_dir(
    base_dir: PathLike,
    prefix: str,
    run_name: Optional[str] = None,
    time_fmt: str = "%Y%m%d-%H%M%S",
) -> Path:
    """
    Create and return an experiment run directory:
      base_dir / f"{prefix}_{timestamp}_{run_name(optional)}"

    Example:
      base_dir="checkpoints/logs"
      prefix="tdar_dm_to_elec"
      -> checkpoints/logs/tdar_dm_to_elec_20260216-153012/

    Args:
      base_dir: where to create run dirs (must be subdir; top-level should exist)
      prefix: run prefix
      run_name: optional suffix name
      time_fmt: timestamp format
    """
    base_dir = Path(base_dir)
    ts = datetime.now().strftime(time_fmt)
    if run_name:
        run_dir = base_dir / f"{prefix}_{ts}_{run_name}"
    else:
        run_dir = base_dir / f"{prefix}_{ts}"
    _ensure_dir(run_dir)
    return run_dir


def log_config(logger: logging.Logger, cfg: dict, title: str = "CONFIG"):
    """
    Pretty log a nested config dict.
    """
    import json
    logger.info(f"==== {title} ====")
    logger.info(json.dumps(cfg, ensure_ascii=False, indent=2))
    logger.info("=" * (len(title) + 10))
