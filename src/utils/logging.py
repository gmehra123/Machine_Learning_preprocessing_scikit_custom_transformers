"""Structured logging utilities for the ML pipeline."""
import logging
import sys
from typing import Dict, Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], stage: str = "") -> None:
    """Log training or evaluation metrics."""
    prefix = f"[{stage}] " if stage else ""
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_drift_warning(logger: logging.Logger, feature: str, drift_type: str, value: float, threshold: float) -> None:
    """Log a drift detection warning."""
    logger.warning(f"DRIFT DETECTED | feature={feature} | type={drift_type} | value={value:.4f} | threshold={threshold:.4f}")


def log_pipeline_stage(logger: logging.Logger, stage: str) -> None:
    """Log pipeline stage entry."""
    logger.info(f"{'='*40}")
    logger.info(f"STAGE: {stage}")
    logger.info(f"{'='*40}")