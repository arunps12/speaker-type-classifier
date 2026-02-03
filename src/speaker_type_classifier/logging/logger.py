import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_log_file_path(
    log_dir: Optional[str] = None,
    run_name: str = "run",
    filename: Optional[str] = None,
) -> Path:
    """
    Create a timestamped log file path.

    Priority:
    - log_dir argument
    - env var LOG_DIR
    - default ./logs

    Example output:
      logs/run_2026-02-03_14-12-10.log
    """
    base_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    _ensure_dir(base_dir)

    if filename:
        return base_dir / filename

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base_dir / f"{run_name}_{ts}.log"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    run_name: str = "run",
    filename: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create (or return) a configured logger with:
    - optional console handler
    - optional file handler
    - safe behavior (no duplicate handlers)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid double-logging through root logger

    # If already configured, return as-is
    if getattr(logger, "_configured", False):
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if file:
        log_path = get_log_file_path(log_dir=log_dir, run_name=run_name, filename=filename)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.log_path = str(log_path)  

    logger._configured = True  
    return logger
